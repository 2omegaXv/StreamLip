"""
FM head training with Auto-AVSR visual features.

Pipeline per step:
  avsr_enc.npy (T,768) → v_down (T_a,768)        [visual condition, pre-extracted]
  smollm2_h.npy (L,960) → h_down (T_a,960)       [text condition, pre-extracted, resampled]
  speaker_emb.npy (256)                           [speaker condition, pre-extracted]
  v_down + h_down + speaker → FMHeadAVSR → CFM loss on Mimi latent

Ablation --no_text_cond: h_down = zeros (visual-only baseline).

Usage:
  python scripts/train_fm_avsr.py --run_name fm_avsr_with_text
  python scripts/train_fm_avsr.py --no_text_cond --run_name fm_avsr_no_text
"""
import argparse, csv, json, math, os, sys, time
from pathlib import Path

import numpy as np

_TMPDIR = Path(__file__).parent.parent / ".tmp"
_TMPDIR.mkdir(exist_ok=True)
os.environ["TMPDIR"] = str(_TMPDIR)

import torch
import torch.multiprocessing
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

torch.multiprocessing.set_sharing_strategy("file_system")
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from streaminlip.v2.fm_head import FMHead as _FMBase, SinusoidalTimeEmb, DiTBlock
from streaminlip.v2.fm_head import masked_mse_loss
import torch.nn as nn

# FM head for Auto-AVSR: vis=768, lm=960, speaker=256 → COND_DIM=1984
class FMHeadAVSR(_FMBase):
    DIM = 512
    def __init__(
        self,
        n_layers=6,
        n_heads=8,
        use_cross_attn=False,
        use_text_token_cross_attn=False,
        extra_cond_dim=0,
        ctc_vocab_size=0,
        ctc_topk=0,
        ctc_token_emb_dim=0,
    ):
        super().__init__(
            n_layers=n_layers,
            n_heads=n_heads,
            use_cross_attn=use_cross_attn,
            use_text_token_cross_attn=use_text_token_cross_attn,
            extra_cond_dim=extra_cond_dim,
            ctc_vocab_size=ctc_vocab_size,
            ctc_topk=ctc_topk,
            ctc_token_emb_dim=ctc_token_emb_dim,
        )
        ctc_cond_dim = ctc_token_emb_dim + ctc_topk if ctc_topk > 0 else 0
        self.cond_proj = nn.Linear(768 + 960 + 256 + extra_cond_dim + ctc_cond_dim, self.DIM)
from streaminlip.fm_avsr_dataset import FMAVSRDataset, collate_fn

try:
    import wandb; _WANDB = True
except ImportError:
    _WANDB = False


def _worker_init(worker_id):
    import os
    os.environ["TMPDIR"] = str(Path(__file__).parent.parent / ".tmp")
    torch.multiprocessing.set_sharing_strategy("file_system")


def explicit_cli_keys(argv=None):
    argv = list(sys.argv if argv is None else argv)
    keys = set()
    for item in argv[1:]:
        if not item.startswith("--"):
            continue
        key = item[2:].split("=", 1)[0].replace("-", "_")
        if key:
            keys.add(key)
    return keys


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config",           default=None,
                   help="Optional YAML config. Values override parser defaults.")
    p.add_argument("--data_root",        default="data/processed")
    p.add_argument("--mimi_path",        default=None,
                   help="Unused by training; accepted so train/eval can share one YAML.")
    p.add_argument("--smollm2_path",     default="pretrained/smollm2-360m",
                   help="Tokenizer path used for word-timestamp text alignment.")
    p.add_argument("--output_dir",       default="runs/fm_avsr")
    p.add_argument("--run_name",         default="fm_avsr_with_text")
    p.add_argument("--no_text_cond",     action="store_true",
                   help="Deprecated alias for --condition_mode video_only.")
    p.add_argument("--condition_mode",
                   choices=["both", "video_only", "text_only", "shuffle_text"],
                   default="both",
                   help="Condition ablation mode for diagnosing video/text contribution.")
    p.add_argument("--text_alignment_mode",
                   choices=["uniform", "word_timestamps"],
                   default="uniform",
                   help="How to align SmolLM2 hidden states to latent frames.")
    p.add_argument("--text_source",
                   choices=["avsr", "text_json"],
                   default="avsr",
                   help="Text source for transcripts and pre-extracted SmolLM2 hidden states.")
    p.add_argument("--ctc_condition_mode",
                   choices=[
                       "none",
                       "logprob",
                       "shuffle_logprob",
                       "summary",
                       "shuffle_summary",
                       "topk",
                       "shuffle_topk",
                   ],
                   default="none",
                   help="Optional Auto-AVSR CTC posterior condition from avsr_enc.")
    p.add_argument("--auto_avsr_ckpt",
                   default="/mnt/pfs/group-jt/zihan.guo/droid/DL-V2A/pretrained/auto_avsr/vsr_trlrs2lrs3vox2avsp_base.pth",
                   help="Checkpoint containing ctc.ctc_lo weights for CTC conditioning.")
    p.add_argument("--ctc_vocab_size", type=int, default=5049,
                   help="Auto-AVSR CTC vocabulary size.")
    p.add_argument("--ctc_topk", type=int, default=4,
                   help="Number of CTC top tokens per frame for topk condition modes.")
    p.add_argument("--ctc_token_emb_dim", type=int, default=32,
                   help="Learned embedding dimension for CTC topk token ids.")
    p.add_argument("--split",            default="pretrain")
    p.add_argument("--clip_list",        default=None,
                   help="Optional file with one processed clip path per line.")
    p.add_argument("--val_clip_list",    default=None,
                   help="Optional held-out clip list for validation.")
    p.add_argument("--resume_ckpt",      default=None,
                   help="Optional checkpoint to resume model weights and step from.")
    p.add_argument("--batch_size",       type=int,   default=1024)
    p.add_argument("--lr",               type=float, default=3e-4)
    p.add_argument("--lr_schedule",      choices=["cosine", "fixed"], default="cosine")
    p.add_argument("--warmup_epochs",    type=float, default=3.0)
    p.add_argument("--max_epochs",       type=int,   default=30)
    p.add_argument("--max_steps",        type=int,   default=0,
                   help="Optional hard cap on optimizer steps; 0 uses max_epochs * steps_per_epoch.")
    p.add_argument("--eval_every",       type=int,   default=500)
    p.add_argument("--save_every",       type=int,   default=1000)
    p.add_argument("--num_workers",      type=int,   default=8)
    p.add_argument("--val_clips",        type=int,   default=500)
    p.add_argument("--crop_ta",          type=int,   default=0,
                   help="Randomly crop each training sample to this many latent frames; 0 disables cropping.")
    p.add_argument("--n_dit_layers",     type=int,   default=6)
    p.add_argument("--use_cross_attn",   action="store_true",
                   help="Insert condition cross-attention in each DiT block.")
    p.add_argument("--use_text_token_cross_attn", action="store_true",
                   help="Use raw SmolLM token sequence as DiT cross-attention K/V.")
    p.add_argument("--loss_fm_weight",   type=float, default=1.0,
                   help="Main FM loss weight. Set to 0 for deterministic recon upper-bound runs.")
    p.add_argument("--lambda_recon",     type=float, default=0.0,
                   help="Auxiliary deterministic latent reconstruction loss weight.")
    p.add_argument("--lambda_sample_recon", type=float, default=0.0,
                   help="Auxiliary sampled endpoint reconstruction loss weight.")
    p.add_argument("--lambda_denoise",  type=float, default=0.0,
                   help="Noisy endpoint denoising loss weight.")
    p.add_argument("--sample_recon_nfe", type=int, default=4,
                   help="Euler steps for sampled endpoint reconstruction loss.")
    p.add_argument("--denoise_t_min",   type=float, default=0.0,
                   help="Minimum denoise time embedding for noisy endpoint training.")
    p.add_argument("--denoise_t_max",   type=float, default=1.0,
                   help="Maximum denoise time embedding for noisy endpoint training.")
    p.add_argument("--eval_sample_nfe",  type=int, default=4,
                   help="Euler steps for validation sampled endpoint metrics.")
    p.add_argument("--no_wandb",         action="store_true")
    p.add_argument("--debug",            action="store_true")
    cli_keys = explicit_cli_keys()
    args = p.parse_args()
    if args.config:
        import yaml
        with open(args.config, "r") as f:
            cfg = yaml.safe_load(f) or {}
        valid = set(vars(args))
        unknown = sorted(set(cfg) - valid)
        if unknown:
            raise ValueError(f"Unknown config keys in {args.config}: {unknown}")
        for k, v in cfg.items():
            if k in cli_keys and k != "config":
                continue
            setattr(args, k, v)
    if args.no_text_cond and "condition_mode" not in cli_keys:
        args.condition_mode = "video_only"
    return args


def get_lr(step, warmup, total, lr):
    if step < warmup: return lr * step / max(warmup, 1)
    t = (step - warmup) / max(total - warmup, 1)
    return lr * 0.5 * (1.0 + math.cos(math.pi * t))


def crop_batch_to_latent_window(enc_np, latent_np, latent_lens, crop_ta, rng=None):
    """Crop each sample to a fixed latent window and the aligned 2x video window."""
    if crop_ta <= 0:
        return enc_np, latent_np, latent_lens
    B, max_ta = latent_np.shape[:2]
    crop_ta = min(crop_ta, max_ta)
    crop_tv = crop_ta * 2
    rng = rng or np.random.default_rng()
    latent_lens = np.asarray(latent_lens, dtype=np.int64)

    enc_crop = np.zeros((B, crop_tv, enc_np.shape[2]), dtype=enc_np.dtype)
    lat_crop = np.zeros((B, crop_ta, latent_np.shape[2]), dtype=latent_np.dtype)
    crop_lens = np.zeros((B,), dtype=np.int64)
    for b in range(B):
        valid_ta = int(latent_lens[b])
        valid_ta = max(1, min(valid_ta, max_ta))
        if valid_ta > crop_ta:
            start_ta = int(rng.integers(0, valid_ta - crop_ta + 1))
            out_ta = crop_ta
        else:
            start_ta = 0
            out_ta = valid_ta
        crop_lens[b] = out_ta
        start_tv = start_ta * 2
        out_tv = min(out_ta * 2, max(0, enc_np.shape[1] - start_tv), crop_tv)
        lat_crop[b, :out_ta] = latent_np[b, start_ta : start_ta + out_ta]
        if out_tv > 0:
            enc_crop[b, :out_tv] = enc_np[b, start_tv : start_tv + out_tv]
    return enc_crop, lat_crop, crop_lens


def aggregate_sample_metrics(pred: torch.Tensor, target: torch.Tensor, lengths: torch.Tensor):
    """Return scalar endpoint metrics over valid latent frames only."""
    lengths = lengths.to(device=pred.device, dtype=torch.long).clamp(min=1, max=pred.shape[1])
    mask = torch.arange(pred.shape[1], device=pred.device).unsqueeze(0) < lengths.unsqueeze(1)
    mask = mask.unsqueeze(-1).expand_as(pred)
    pred_v = pred.float()[mask]
    target_v = target.float()[mask]
    diff = pred_v - target_v
    mse = diff.pow(2).mean()
    mae = diff.abs().mean()
    rel_l2 = torch.linalg.vector_norm(diff) / torch.linalg.vector_norm(target_v).clamp_min(1e-8)
    pred_c = pred_v - pred_v.mean()
    target_c = target_v - target_v.mean()
    corr = (pred_c * target_c).mean() / (
        pred_c.pow(2).mean().sqrt() * target_c.pow(2).mean().sqrt()
    ).clamp_min(1e-8)
    return {
        "mse": float(mse.detach().cpu()),
        "mae": float(mae.detach().cpu()),
        "corr": float(corr.detach().cpu()),
        "rel_l2": float(rel_l2.detach().cpu()),
    }


def combine_training_losses(
    loss_fm: torch.Tensor,
    loss_recon: torch.Tensor,
    loss_sample_recon: torch.Tensor,
    loss_denoise: torch.Tensor,
    loss_fm_weight: float,
    lambda_recon: float,
    lambda_sample_recon: float,
    lambda_denoise: float,
) -> torch.Tensor:
    return (
        loss_fm_weight * loss_fm
        + lambda_recon * loss_recon
        + lambda_sample_recon * loss_sample_recon
        + lambda_denoise * loss_denoise
    )


def resample_h_lm(h_lm_np, lens_L, T_a, device):
    """Resample pre-extracted SmolLM2 hidden states to T_a latent frames.

    Args:
        h_lm_np: numpy (B, max_L, 960)
        lens_L:  numpy (B,) actual token lengths per sample
        T_a:     target time steps
        device:  torch device
    Returns:
        (B, T_a, 960) bfloat16 tensor
    """
    B, _, D = h_lm_np.shape
    h_lm_t = torch.from_numpy(h_lm_np).to(device, dtype=torch.bfloat16)  # (B, max_L, D)
    lens_t  = torch.from_numpy(lens_L.astype("int64")).to(device)         # (B,)

    # Vectorized index: idx[b, j] = clamp(j * lens_L[b] // T_a, 0, lens_L[b]-1)
    arange  = torch.arange(T_a, device=device, dtype=torch.int64).unsqueeze(0)  # (1, T_a)
    lens_u  = lens_t.unsqueeze(1)                                                # (B, 1)
    idx     = (arange * lens_u // max(T_a, 1)).clamp(min=0)                      # (B, T_a)
    idx     = torch.minimum(idx, lens_u - 1)                                     # cap to L-1

    # Single batched gather: h_lm_t[b, idx[b, j], :] → (B, T_a, D)
    idx_exp = idx.unsqueeze(-1).expand(-1, -1, D)     # (B, T_a, D)
    return torch.gather(h_lm_t, 1, idx_exp)           # (B, T_a, D)


def gather_h_lm_by_lm_idx(h_lm_np, lens_L, lm_idx_np, T_a, device):
    """Gather hidden states by per-latent LM indices from timestamp alignment."""
    B, _, D = h_lm_np.shape
    h_lm_t = torch.from_numpy(h_lm_np).to(device, dtype=torch.bfloat16)
    lens_t = torch.from_numpy(lens_L.astype("int64")).to(device)
    idx = torch.from_numpy(lm_idx_np[:, :T_a].astype("int64")).to(device)
    idx = idx.clamp(min=0)
    idx = torch.minimum(idx, (lens_t - 1).unsqueeze(1))
    idx_exp = idx.unsqueeze(-1).expand(-1, -1, D)
    return torch.gather(h_lm_t, 1, idx_exp)


def prepare_conditions(
    batch,
    device,
    condition_mode="both",
    text_perm=None,
    text_alignment_mode="uniform",
    ctc_condition_mode="none",
):
    enc = torch.from_numpy(batch["enc"]).to(device, dtype=torch.bfloat16)
    lat_gt = torch.from_numpy(batch["latent"]).to(device, dtype=torch.float32)
    lat_lens = torch.from_numpy(batch["latent_lens"]).to(device)
    spk = torch.from_numpy(batch["speaker"]).to(device, dtype=torch.bfloat16)
    B, T_a = lat_gt.shape[:2]

    v_down = enc[:, ::2, :][:, :T_a, :]
    if v_down.shape[1] < T_a:
        pad = torch.zeros(B, T_a - v_down.shape[1], 768,
                          device=device, dtype=torch.bfloat16)
        v_down = torch.cat([v_down, pad], 1)

    if condition_mode == "text_only":
        v_down = torch.zeros_like(v_down)

    text_tokens = None
    text_token_mask = None
    if condition_mode == "video_only" or batch["h_lm"] is None:
        h_down = torch.zeros(B, T_a, 960, device=device, dtype=torch.bfloat16)
    else:
        h_lm_np = batch["h_lm"]
        lens_L = batch["lens_L"]
        if condition_mode == "shuffle_text":
            if text_perm is None:
                text_perm = torch.randperm(B, device=device).cpu().numpy()
            h_lm_np = h_lm_np[text_perm]
            lens_L = lens_L[text_perm]
        if text_alignment_mode == "word_timestamps":
            if batch.get("lm_idx") is None:
                raise ValueError("word_timestamps alignment requires batch['lm_idx']")
            lm_idx_np = batch["lm_idx"]
            if condition_mode == "shuffle_text":
                lm_idx_np = lm_idx_np[text_perm]
            h_down = gather_h_lm_by_lm_idx(h_lm_np, lens_L, lm_idx_np, T_a, device)
        else:
            h_down = resample_h_lm(h_lm_np, lens_L, T_a, device)
        text_tokens = torch.from_numpy(h_lm_np).to(device, dtype=torch.bfloat16)
        lens_t = torch.from_numpy(lens_L.astype("int64")).to(device)
        pos = torch.arange(text_tokens.shape[1], device=device).unsqueeze(0)
        text_token_mask = pos < lens_t.unsqueeze(1)
    extra_cond = None
    ctc_topk_ids = None
    ctc_topk_probs = None
    if ctc_condition_mode in {"logprob", "shuffle_logprob", "summary", "shuffle_summary"}:
        if batch.get("ctc_cond") is None:
            raise ValueError("CTC condition requires batch['ctc_cond']")
        raw_ctc = batch["ctc_cond"]
        if isinstance(raw_ctc, torch.Tensor):
            ctc = raw_ctc.to(device=device, dtype=torch.bfloat16)
        else:
            ctc = torch.from_numpy(raw_ctc).to(device, dtype=torch.bfloat16)
        extra_cond = ctc[:, ::2, :][:, :T_a, :]
        if extra_cond.shape[1] < T_a:
            pad = torch.zeros(
                B, T_a - extra_cond.shape[1], extra_cond.shape[2],
                device=device, dtype=torch.bfloat16,
            )
            extra_cond = torch.cat([extra_cond, pad], dim=1)
    elif ctc_condition_mode in {"topk", "shuffle_topk"}:
        if batch.get("ctc_topk_ids") is None or batch.get("ctc_topk_probs") is None:
            raise ValueError("CTC top-k condition requires ids and probs")
        raw_ids = batch["ctc_topk_ids"]
        raw_probs = batch["ctc_topk_probs"]
        if isinstance(raw_ids, torch.Tensor):
            ids = raw_ids.to(device=device, dtype=torch.long)
        else:
            ids = torch.from_numpy(raw_ids).to(device=device, dtype=torch.long)
        if isinstance(raw_probs, torch.Tensor):
            probs = raw_probs.to(device=device, dtype=torch.bfloat16)
        else:
            probs = torch.from_numpy(raw_probs).to(device=device, dtype=torch.bfloat16)
        ctc_topk_ids = ids[:, ::2, :][:, :T_a, :]
        ctc_topk_probs = probs[:, ::2, :][:, :T_a, :]
        if ctc_topk_ids.shape[1] < T_a:
            pad_ids = torch.zeros(
                B, T_a - ctc_topk_ids.shape[1], ctc_topk_ids.shape[2],
                device=device, dtype=torch.long,
            )
            pad_probs = torch.zeros(
                B, T_a - ctc_topk_probs.shape[1], ctc_topk_probs.shape[2],
                device=device, dtype=torch.bfloat16,
            )
            ctc_topk_ids = torch.cat([ctc_topk_ids, pad_ids], dim=1)
            ctc_topk_probs = torch.cat([ctc_topk_probs, pad_probs], dim=1)
    return (
        v_down,
        h_down,
        spk,
        lat_gt,
        lat_lens,
        text_tokens,
        text_token_mask,
        extra_cond,
        ctc_topk_ids,
        ctc_topk_probs,
    )


def load_ctc_head(ckpt_path: str, vocab_size: int = 5049):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ckpt.get("model_state_dict", ckpt)
    head = nn.Linear(768, vocab_size)
    head.weight.data.copy_(sd["ctc.ctc_lo.weight"].float())
    head.bias.data.copy_(sd["ctc.ctc_lo.bias"].float())
    for p in head.parameters():
        p.requires_grad_(False)
    return head.eval()


def ctc_extra_dim(mode: str, vocab_size: int) -> int:
    if mode == "none":
        return 0
    if mode in {"summary", "shuffle_summary"}:
        return 6
    if mode in {"topk", "shuffle_topk"}:
        return 0
    return vocab_size


def ctc_topk_dim(mode: str, topk: int) -> int:
    return topk if mode in {"topk", "shuffle_topk"} else 0


def attach_ctc_condition(batch, ctc_head, device, mode: str, vocab_size: int, topk: int = 4):
    if ctc_head is None:
        return batch
    enc = torch.from_numpy(batch["enc"]).to(device, dtype=torch.float32)
    with torch.no_grad():
        logits = ctc_head(enc)
        if mode in {"logprob", "shuffle_logprob"}:
            cond = F.log_softmax(logits, dim=-1).clamp(min=-20.0, max=0.0)
            if mode == "shuffle_logprob":
                perm = torch.randperm(cond.shape[0], device=device)
                cond = cond[perm]
            batch["ctc_cond"] = cond.detach()
        elif mode in {"summary", "shuffle_summary"}:
            prob = F.softmax(logits.float(), dim=-1)
            top_prob, top_idx = torch.topk(prob, k=2, dim=-1)
            entropy = -(prob * prob.clamp_min(1e-8).log()).sum(dim=-1, keepdim=True)
            entropy = entropy / math.log(vocab_size)
            cond = torch.cat([
                prob[..., :1],
                top_prob[..., :1],
                top_prob[..., 1:2],
                top_idx[..., :1].float() / max(vocab_size - 1, 1),
                top_idx[..., 1:2].float() / max(vocab_size - 1, 1),
                entropy,
            ], dim=-1)
            if mode == "shuffle_summary":
                perm = torch.randperm(cond.shape[0], device=device)
                cond = cond[perm]
            batch["ctc_cond"] = cond.detach()
        elif mode in {"topk", "shuffle_topk"}:
            prob = F.softmax(logits.float(), dim=-1)
            top_probs, top_ids = torch.topk(prob, k=topk, dim=-1)
            if mode == "shuffle_topk":
                perm = torch.randperm(top_ids.shape[0], device=device)
                top_ids = top_ids[perm]
                top_probs = top_probs[perm]
            batch["ctc_topk_ids"] = top_ids.detach()
            batch["ctc_topk_probs"] = top_probs.detach()
    return batch


def main():
    args   = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path(args.output_dir) / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "metrics.csv"
    val_metrics_path = out_dir / "val_metrics.csv"
    config_path = out_dir / "config.json"
    if not config_path.exists():
        config_path.write_text(json.dumps(vars(args), indent=2, sort_keys=True) + "\n")
    metrics_exists = metrics_path.exists() and metrics_path.stat().st_size > 0
    metrics_f = metrics_path.open("a", newline="")
    metrics = csv.DictWriter(metrics_f, fieldnames=[
        "step", "epoch", "loss_fm", "loss_recon", "loss_total", "lr",
        "loss_sample_recon", "loss_denoise", "seconds_per_step", "elapsed_seconds"
    ])
    if not metrics_exists:
        metrics.writeheader()
        metrics_f.flush()
    val_metrics_exists = val_metrics_path.exists() and val_metrics_path.stat().st_size > 0
    val_metrics_f = val_metrics_path.open("a", newline="")
    val_metrics = csv.DictWriter(val_metrics_f, fieldnames=[
        "step", "epoch", "val_loss_fm",
        "val_sample_mse", "val_sample_mae", "val_sample_corr", "val_sample_rel_l2",
        "val_recon_mse", "val_recon_mae", "val_recon_corr", "val_recon_rel_l2",
        "val_denoise_mse", "val_denoise_mae", "val_denoise_corr", "val_denoise_rel_l2",
        "train_sample_mse", "train_sample_mae", "train_sample_corr", "train_sample_rel_l2",
        "train_recon_mse", "train_recon_mae", "train_recon_corr", "train_recon_rel_l2",
        "train_denoise_mse", "train_denoise_mae", "train_denoise_corr", "train_denoise_rel_l2",
        "n_batches", "elapsed_seconds"
    ])
    if not val_metrics_exists:
        val_metrics.writeheader()
        val_metrics_f.flush()
    cond   = args.condition_mode
    print(
        f"FM AVSR | cond={cond} | text_align={args.text_alignment_mode} "
        f"| ctc={args.ctc_condition_mode} | epochs={args.max_epochs}"
    )

    use_wandb = _WANDB and not args.no_wandb
    if use_wandb:
        wandb.init(entity="gzh-thu", project="StreamLip",
                   name=args.run_name, config=vars(args))

    # ── FM head (trainable) ───────────────────────────────────────────────────
    fm = FMHeadAVSR(
        n_layers=args.n_dit_layers,
        use_cross_attn=args.use_cross_attn,
        use_text_token_cross_attn=args.use_text_token_cross_attn,
        extra_cond_dim=ctc_extra_dim(args.ctc_condition_mode, args.ctc_vocab_size),
        ctc_vocab_size=args.ctc_vocab_size,
        ctc_topk=ctc_topk_dim(args.ctc_condition_mode, args.ctc_topk),
        ctc_token_emb_dim=args.ctc_token_emb_dim,
    ).to(device).bfloat16()
    resume_step = 0
    if args.resume_ckpt:
        ckpt = torch.load(args.resume_ckpt, map_location="cpu", weights_only=False)
        fm.load_state_dict(ckpt["fm_head"])
        resume_step = int(ckpt.get("step", 0))
        print(f"  resumed {args.resume_ckpt} at step={resume_step}")
    n_train = sum(p.numel() for p in fm.parameters())
    print(f"FM head: {n_train/1e6:.1f}M params (all trainable)")

    ctc_head = None
    if args.ctc_condition_mode != "none":
        ctc_head = load_ctc_head(args.auto_avsr_ckpt, args.ctc_vocab_size).to(device)

    # ── Dataset ───────────────────────────────────────────────────────────────
    limit = 32 if args.debug else None
    if args.debug: args.batch_size = min(args.batch_size, 4)
    tokenizer = None
    if args.text_alignment_mode == "word_timestamps":
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.smollm2_path, local_files_only=True)
    ds = FMAVSRDataset(
        args.data_root, args.split, subset="train",
        limit=limit, clip_list=args.clip_list,
        tokenizer=tokenizer, text_alignment_mode=args.text_alignment_mode,
        text_source=args.text_source,
    )
    if args.val_clip_list:
        train_ds = ds
        val_ds = FMAVSRDataset(
            args.data_root, args.split, subset="train",
            limit=limit, clip_list=args.val_clip_list,
            tokenizer=tokenizer, text_alignment_mode=args.text_alignment_mode,
            text_source=args.text_source,
        )
        train_n = len(train_ds)
        val_n = len(val_ds)
    else:
        val_n = min(args.val_clips, len(ds))
        train_n = len(ds) - val_n
    if args.val_clip_list:
        pass
    elif val_n > 0:
        train_ds, val_ds = random_split(ds, [train_n, val_n],
                                        generator=torch.Generator().manual_seed(42))
    else:
        train_ds, val_ds = ds, None
    print(f"train={train_n}  val={val_n}")

    steps_per_epoch = max(1, train_n // args.batch_size)
    max_steps    = args.max_steps if args.max_steps > 0 else steps_per_epoch * args.max_epochs
    warmup_steps = max(1, int(steps_per_epoch * args.warmup_epochs))
    print(f"  steps/epoch={steps_per_epoch} | warmup={warmup_steps} | max={max_steps}")

    loader_kw = dict(batch_size=args.batch_size, collate_fn=collate_fn,
                     pin_memory=False, drop_last=True, num_workers=args.num_workers)
    if args.num_workers > 0:
        loader_kw.update(dict(multiprocessing_context="fork",
                              worker_init_fn=_worker_init, persistent_workers=True,
                              prefetch_factor=2))
    train_loader = DataLoader(train_ds, shuffle=True,  **loader_kw)
    val_loader = None
    if val_ds is not None:
        val_loader = DataLoader(val_ds,   shuffle=False,
                                batch_size=args.batch_size, num_workers=0,
                                collate_fn=collate_fn, pin_memory=False)

    # ── Optimizer ─────────────────────────────────────────────────────────────
    opt = torch.optim.AdamW(fm.parameters(), lr=args.lr,
                            betas=(0.9, 0.95), weight_decay=0.1)

    # ── Training loop ─────────────────────────────────────────────────────────
    step = resume_step; t_log = time.time(); t_start = time.time()
    last_save_loss = None
    last_save_recon = None
    last_save_sample_recon = None
    last_save_denoise = None
    last_save_total = None
    last_save_lr = None
    fm.train()
    crop_rng = np.random.default_rng(42)

    try:
        for epoch in range(100_000):
            for batch in train_loader:
                if ctc_head is not None:
                    batch = attach_ctc_condition(
                        batch,
                        ctc_head,
                        device,
                        args.ctc_condition_mode,
                        args.ctc_vocab_size,
                        args.ctc_topk,
                    )
                if args.crop_ta > 0:
                    batch["enc"], batch["latent"], batch["latent_lens"] = crop_batch_to_latent_window(
                        batch["enc"], batch["latent"], batch["latent_lens"], args.crop_ta, crop_rng
                    )
                (
                    v_down,
                    h_down,
                    spk,
                    lat_gt,
                    lat_lens,
                    text_tokens,
                    text_mask,
                    extra_cond,
                    ctc_ids,
                    ctc_probs,
                ) = prepare_conditions(
                    batch, device, condition_mode=args.condition_mode,
                    text_alignment_mode=args.text_alignment_mode,
                    ctc_condition_mode=args.ctc_condition_mode,
                )

                # FM loss
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    if args.loss_fm_weight > 0:
                        loss_fm = fm.forward_train(
                            v_down, h_down, spk, lat_gt,
                            lengths=lat_lens,
                            text_tokens=text_tokens,
                            text_token_mask=text_mask,
                            extra_cond=extra_cond,
                            ctc_topk_ids=ctc_ids,
                            ctc_topk_probs=ctc_probs,
                        )
                    else:
                        loss_fm = lat_gt.new_zeros(())
                    if args.lambda_recon > 0:
                        pred_recon = fm.reconstruct_from_cond(
                            v_down, h_down, spk,
                            text_tokens=text_tokens,
                            text_token_mask=text_mask,
                            extra_cond=extra_cond,
                            ctc_topk_ids=ctc_ids,
                            ctc_topk_probs=ctc_probs,
                        )
                        loss_recon = masked_mse_loss(pred_recon, lat_gt, lat_lens)
                    else:
                        loss_recon = loss_fm.new_zeros(())
                    if args.lambda_sample_recon > 0:
                        pred_sample = fm.sample(
                            v_down, h_down, spk,
                            nfe=args.sample_recon_nfe,
                            text_tokens=text_tokens,
                            text_token_mask=text_mask,
                            extra_cond=extra_cond,
                            ctc_topk_ids=ctc_ids,
                            ctc_topk_probs=ctc_probs,
                        )
                        loss_sample_recon = masked_mse_loss(pred_sample, lat_gt, lat_lens)
                    else:
                        loss_sample_recon = loss_fm.new_zeros(())
                    if args.lambda_denoise > 0:
                        noise = torch.randn_like(lat_gt).to(dtype=torch.bfloat16)
                        denoise_t = torch.empty(
                            lat_gt.shape[0], device=lat_gt.device, dtype=torch.bfloat16
                        ).uniform_(args.denoise_t_min, args.denoise_t_max)
                        pred_denoise = fm.denoise_from_noise(
                            v_down, h_down, spk, noise, denoise_t,
                            text_tokens=text_tokens,
                            text_token_mask=text_mask,
                            extra_cond=extra_cond,
                            ctc_topk_ids=ctc_ids,
                            ctc_topk_probs=ctc_probs,
                        )
                        loss_denoise = masked_mse_loss(pred_denoise, lat_gt, lat_lens)
                    else:
                        loss_denoise = loss_fm.new_zeros(())
                    loss = combine_training_losses(
                        loss_fm,
                        loss_recon,
                        loss_sample_recon,
                        loss_denoise,
                        args.loss_fm_weight,
                        args.lambda_recon,
                        args.lambda_sample_recon,
                        args.lambda_denoise,
                    )

                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(fm.parameters(), 1.0)
                lr = args.lr if args.lr_schedule == "fixed" else get_lr(
                    step, warmup_steps, max_steps, args.lr
                )
                for pg in opt.param_groups: pg["lr"] = lr
                opt.step()
                step += 1

                elapsed = time.time() - t_start
                metrics.writerow({
                    "step": step,
                    "epoch": epoch,
                    "loss_fm": f"{loss_fm.item():.8f}",
                    "loss_recon": f"{loss_recon.item():.8f}",
                    "loss_sample_recon": f"{loss_sample_recon.item():.8f}",
                    "loss_denoise": f"{loss_denoise.item():.8f}",
                    "loss_total": f"{loss.item():.8f}",
                    "lr": f"{lr:.10e}",
                    "seconds_per_step": "",
                    "elapsed_seconds": f"{elapsed:.4f}",
                })
                metrics_f.flush()
                last_save_loss = loss_fm.item()
                last_save_recon = loss_recon.item()
                last_save_sample_recon = loss_sample_recon.item()
                last_save_denoise = loss_denoise.item()
                last_save_total = loss.item()
                last_save_lr = lr

                if step % 10 == 0:
                    sps = (time.time() - t_log) / 10; t_log = time.time()
                    eta = f"{(max_steps-step)*sps/3600:.1f}h"
                    print(f"step {step:6d}/{max_steps} | fm {loss_fm.item():.4f} | "
                          f"recon {loss_recon.item():.4f} | "
                          f"sample {loss_sample_recon.item():.4f} | "
                          f"denoise {loss_denoise.item():.4f} | "
                          f"total {loss.item():.4f} | "
                          f"lr {lr:.2e} | {sps:.2f}s/step | eta {eta}", flush=True)
                    if use_wandb:
                        wandb.log({
                            "train/loss_fm": loss_fm.item(),
                            "train/loss_recon": loss_recon.item(),
                            "train/loss_sample_recon": loss_sample_recon.item(),
                            "train/loss_denoise": loss_denoise.item(),
                            "train/loss_total": loss.item(),
                            "train/lr": lr,
                        }, step=step)

                if val_loader is not None and step % args.eval_every == 0:
                    fm.eval()
                    total = 0.0; n = 0
                    val_sample = {"mse": [], "mae": [], "corr": [], "rel_l2": []}
                    val_recon = {"mse": [], "mae": [], "corr": [], "rel_l2": []}
                    val_denoise = {"mse": [], "mae": [], "corr": [], "rel_l2": []}
                    with torch.no_grad():
                        for vb in val_loader:
                            if ctc_head is not None:
                                vb = attach_ctc_condition(
                                    vb,
                                    ctc_head,
                                    device,
                                    args.ctc_condition_mode,
                                    args.ctc_vocab_size,
                                    args.ctc_topk,
                                )
                            (
                                vd_v,
                                hd_v,
                                spk_v,
                                lat_v,
                                lat_v_lens,
                                text_tokens_v,
                                text_mask_v,
                                extra_v,
                                ctc_ids_v,
                                ctc_probs_v,
                            ) = prepare_conditions(
                                vb, device, condition_mode=args.condition_mode,
                                text_alignment_mode=args.text_alignment_mode,
                                ctc_condition_mode=args.ctc_condition_mode,
                            )
                            if args.loss_fm_weight > 0:
                                total += fm.forward_train(
                                    vd_v, hd_v, spk_v, lat_v, lengths=lat_v_lens,
                                    text_tokens=text_tokens_v,
                                    text_token_mask=text_mask_v,
                                    extra_cond=extra_v,
                                    ctc_topk_ids=ctc_ids_v,
                                    ctc_topk_probs=ctc_probs_v,
                                ).item()
                            pred_recon_v = fm.reconstruct_from_cond(
                                vd_v, hd_v, spk_v,
                                text_tokens=text_tokens_v,
                                text_token_mask=text_mask_v,
                                extra_cond=extra_v,
                                ctc_topk_ids=ctc_ids_v,
                                ctc_topk_probs=ctc_probs_v,
                            )
                            recon_metrics = aggregate_sample_metrics(pred_recon_v, lat_v, lat_v_lens)
                            for k, v in recon_metrics.items():
                                val_recon[k].append(v)
                            if args.lambda_denoise > 0:
                                noise_v = torch.randn_like(lat_v).to(dtype=torch.bfloat16)
                                denoise_t_v = torch.full(
                                    (lat_v.shape[0],),
                                    args.denoise_t_min,
                                    device=lat_v.device,
                                    dtype=torch.bfloat16,
                                )
                                pred_denoise_v = fm.denoise_from_noise(
                                    vd_v, hd_v, spk_v, noise_v, denoise_t_v,
                                    text_tokens=text_tokens_v,
                                    text_token_mask=text_mask_v,
                                    extra_cond=extra_v,
                                    ctc_topk_ids=ctc_ids_v,
                                    ctc_topk_probs=ctc_probs_v,
                                )
                                denoise_metrics = aggregate_sample_metrics(
                                    pred_denoise_v, lat_v, lat_v_lens
                                )
                                for k, v in denoise_metrics.items():
                                    val_denoise[k].append(v)
                            if args.eval_sample_nfe > 0:
                                pred_v = fm.forward_inference(
                                    vd_v, hd_v, spk_v,
                                    nfe=args.eval_sample_nfe,
                                    text_tokens=text_tokens_v,
                                    text_token_mask=text_mask_v,
                                    extra_cond=extra_v,
                                    ctc_topk_ids=ctc_ids_v,
                                    ctc_topk_probs=ctc_probs_v,
                                )
                                sample_metrics = aggregate_sample_metrics(pred_v, lat_v, lat_v_lens)
                                for k, v in sample_metrics.items():
                                    val_sample[k].append(v)
                            n += 1
                            if n >= 10: break
                    val_fm = total / max(n, 1) if args.loss_fm_weight > 0 else 0.0
                    val_sample_mean = {
                        k: sum(v) / max(len(v), 1) for k, v in val_sample.items()
                    }
                    val_recon_mean = {
                        k: sum(v) / max(len(v), 1) for k, v in val_recon.items()
                    }
                    val_denoise_mean = {
                        k: sum(v) / max(len(v), 1) for k, v in val_denoise.items()
                    }
                    with torch.no_grad():
                        pred_train_recon = fm.reconstruct_from_cond(
                            v_down, h_down, spk,
                            text_tokens=text_tokens,
                            text_token_mask=text_mask,
                            extra_cond=extra_cond,
                            ctc_topk_ids=ctc_ids,
                            ctc_topk_probs=ctc_probs,
                        )
                        train_recon = aggregate_sample_metrics(pred_train_recon, lat_gt, lat_lens)
                        if args.lambda_denoise > 0:
                            noise_train = torch.randn_like(lat_gt).to(dtype=torch.bfloat16)
                            denoise_t_train = torch.full(
                                (lat_gt.shape[0],),
                                args.denoise_t_min,
                                device=lat_gt.device,
                                dtype=torch.bfloat16,
                            )
                            pred_train_denoise = fm.denoise_from_noise(
                                v_down, h_down, spk, noise_train, denoise_t_train,
                                text_tokens=text_tokens,
                                text_token_mask=text_mask,
                                extra_cond=extra_cond,
                                ctc_topk_ids=ctc_ids,
                                ctc_topk_probs=ctc_probs,
                            )
                            train_denoise = aggregate_sample_metrics(
                                pred_train_denoise, lat_gt, lat_lens
                            )
                        else:
                            train_denoise = {"mse": 0.0, "mae": 0.0, "corr": 0.0, "rel_l2": 0.0}
                        if args.eval_sample_nfe > 0:
                            pred_train = fm.forward_inference(
                                v_down, h_down, spk,
                                nfe=args.eval_sample_nfe,
                                text_tokens=text_tokens,
                                text_token_mask=text_mask,
                                extra_cond=extra_cond,
                                ctc_topk_ids=ctc_ids,
                                ctc_topk_probs=ctc_probs,
                            )
                            train_sample = aggregate_sample_metrics(pred_train, lat_gt, lat_lens)
                        else:
                            train_sample = {"mse": 0.0, "mae": 0.0, "corr": 0.0, "rel_l2": 0.0}
                    print(
                        f"  [val] loss_fm {val_fm:.4f} | "
                        f"sample corr {val_sample_mean['corr']:.4f} | "
                        f"recon corr {val_recon_mean['corr']:.4f} | "
                        f"denoise corr {val_denoise_mean['corr']:.4f} | "
                        f"train denoise corr {train_denoise['corr']:.4f}",
                        flush=True,
                    )
                    val_metrics.writerow({
                        "step": step,
                        "epoch": epoch,
                        "val_loss_fm": f"{val_fm:.8f}",
                        "val_sample_mse": f"{val_sample_mean['mse']:.8f}",
                        "val_sample_mae": f"{val_sample_mean['mae']:.8f}",
                        "val_sample_corr": f"{val_sample_mean['corr']:.8f}",
                        "val_sample_rel_l2": f"{val_sample_mean['rel_l2']:.8f}",
                        "val_recon_mse": f"{val_recon_mean['mse']:.8f}",
                        "val_recon_mae": f"{val_recon_mean['mae']:.8f}",
                        "val_recon_corr": f"{val_recon_mean['corr']:.8f}",
                        "val_recon_rel_l2": f"{val_recon_mean['rel_l2']:.8f}",
                        "val_denoise_mse": f"{val_denoise_mean['mse']:.8f}",
                        "val_denoise_mae": f"{val_denoise_mean['mae']:.8f}",
                        "val_denoise_corr": f"{val_denoise_mean['corr']:.8f}",
                        "val_denoise_rel_l2": f"{val_denoise_mean['rel_l2']:.8f}",
                        "train_sample_mse": f"{train_sample['mse']:.8f}",
                        "train_sample_mae": f"{train_sample['mae']:.8f}",
                        "train_sample_corr": f"{train_sample['corr']:.8f}",
                        "train_sample_rel_l2": f"{train_sample['rel_l2']:.8f}",
                        "train_recon_mse": f"{train_recon['mse']:.8f}",
                        "train_recon_mae": f"{train_recon['mae']:.8f}",
                        "train_recon_corr": f"{train_recon['corr']:.8f}",
                        "train_recon_rel_l2": f"{train_recon['rel_l2']:.8f}",
                        "train_denoise_mse": f"{train_denoise['mse']:.8f}",
                        "train_denoise_mae": f"{train_denoise['mae']:.8f}",
                        "train_denoise_corr": f"{train_denoise['corr']:.8f}",
                        "train_denoise_rel_l2": f"{train_denoise['rel_l2']:.8f}",
                        "n_batches": n,
                        "elapsed_seconds": f"{time.time() - t_start:.4f}",
                    })
                    val_metrics_f.flush()
                    if use_wandb: wandb.log({"val/loss_fm": val_fm}, step=step)
                    fm.train()

                if step % args.save_every == 0 or step >= max_steps:
                    path = out_dir / f"step_{step:06d}.pt"
                    torch.save({
                        "step": step,
                        "fm_head": fm.state_dict(),
                        "loss_fm": last_save_loss,
                        "loss_recon": last_save_recon,
                        "loss_sample_recon": last_save_sample_recon,
                        "loss_denoise": last_save_denoise,
                        "loss_total": last_save_total,
                        "lr": last_save_lr,
                        "args": vars(args),
                    }, str(path))
                    print(f"  [save] {path}", flush=True)

                if args.debug and step >= 3: print("Debug done."); return
                if step >= max_steps: print("Training complete."); return
    finally:
        metrics_f.close()
        val_metrics_f.close()


if __name__ == "__main__":
    main()
