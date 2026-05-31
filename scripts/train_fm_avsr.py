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

_TMPDIR = Path(__file__).parent.parent / ".tmp"
_TMPDIR.mkdir(exist_ok=True)
os.environ["TMPDIR"] = str(_TMPDIR)

import torch
import torch.multiprocessing
from torch.utils.data import DataLoader, random_split

torch.multiprocessing.set_sharing_strategy("file_system")
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from streaminlip.v2.fm_head import FMHead as _FMBase, SinusoidalTimeEmb, DiTBlock
import torch.nn as nn

# FM head for Auto-AVSR: vis=768, lm=960, speaker=256 → COND_DIM=1984
class FMHeadAVSR(_FMBase):
    DIM = 512
    def __init__(self, n_layers=6, n_heads=8, use_cross_attn=False):
        nn.Module.__init__(self)
        self.use_cross_attn = use_cross_attn
        COND_DIM = 768 + 960 + 256  # avsr_enc + smollm2 h_lm + speaker = 1984
        self.cond_proj  = nn.Linear(COND_DIM, self.DIM)
        self.cond_token_proj = nn.Linear(self.DIM, self.DIM)
        self.time_emb   = SinusoidalTimeEmb(self.DIM)
        self.blocks     = nn.ModuleList([
            DiTBlock(self.DIM, n_heads, use_cross_attn=use_cross_attn)
            for _ in range(n_layers)
        ])
        self.final_norm = nn.LayerNorm(self.DIM)
        self.final_proj = nn.Linear(self.DIM, self.DIM)
        nn.init.zeros_(self.final_proj.weight)
        nn.init.zeros_(self.final_proj.bias)
from streaminlip.fm_avsr_dataset import FMAVSRDataset, collate_fn

try:
    import wandb; _WANDB = True
except ImportError:
    _WANDB = False


def _worker_init(worker_id):
    import os
    os.environ["TMPDIR"] = str(Path(__file__).parent.parent / ".tmp")
    torch.multiprocessing.set_sharing_strategy("file_system")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config",           default=None,
                   help="Optional YAML config. Values override parser defaults.")
    p.add_argument("--data_root",        default="data/processed")
    p.add_argument("--mimi_path",        default=None,
                   help="Unused by training; accepted so train/eval can share one YAML.")
    p.add_argument("--output_dir",       default="runs/fm_avsr")
    p.add_argument("--run_name",         default="fm_avsr_with_text")
    p.add_argument("--no_text_cond",     action="store_true")
    p.add_argument("--split",            default="pretrain")
    p.add_argument("--clip_list",        default=None,
                   help="Optional file with one processed clip path per line.")
    p.add_argument("--batch_size",       type=int,   default=1024)
    p.add_argument("--lr",               type=float, default=3e-4)
    p.add_argument("--warmup_epochs",    type=float, default=3.0)
    p.add_argument("--max_epochs",       type=int,   default=30)
    p.add_argument("--eval_every",       type=int,   default=500)
    p.add_argument("--save_every",       type=int,   default=1000)
    p.add_argument("--num_workers",      type=int,   default=8)
    p.add_argument("--val_clips",        type=int,   default=500)
    p.add_argument("--n_dit_layers",     type=int,   default=6)
    p.add_argument("--use_cross_attn",   action="store_true",
                   help="Insert condition cross-attention in each DiT block.")
    p.add_argument("--lambda_recon",     type=float, default=0.0,
                   help="Auxiliary deterministic latent reconstruction loss weight.")
    p.add_argument("--lambda_sample_recon", type=float, default=0.0,
                   help="Auxiliary sampled endpoint reconstruction loss weight.")
    p.add_argument("--sample_recon_nfe", type=int, default=4,
                   help="Euler steps for sampled endpoint reconstruction loss.")
    p.add_argument("--no_wandb",         action="store_true")
    p.add_argument("--debug",            action="store_true")
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
            setattr(args, k, v)
    return args


def get_lr(step, warmup, total, lr):
    if step < warmup: return lr * step / max(warmup, 1)
    t = (step - warmup) / max(total - warmup, 1)
    return lr * 0.5 * (1.0 + math.cos(math.pi * t))


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
        "loss_sample_recon", "seconds_per_step", "elapsed_seconds"
    ])
    if not metrics_exists:
        metrics.writeheader()
        metrics_f.flush()
    val_metrics_exists = val_metrics_path.exists() and val_metrics_path.stat().st_size > 0
    val_metrics_f = val_metrics_path.open("a", newline="")
    val_metrics = csv.DictWriter(val_metrics_f, fieldnames=[
        "step", "epoch", "val_loss_fm", "n_batches", "elapsed_seconds"
    ])
    if not val_metrics_exists:
        val_metrics.writeheader()
        val_metrics_f.flush()
    cond   = "no_text" if args.no_text_cond else "with_text"
    print(f"FM AVSR | cond={cond} | epochs={args.max_epochs}")

    use_wandb = _WANDB and not args.no_wandb
    if use_wandb:
        wandb.init(entity="gzh-thu", project="StreamLip",
                   name=args.run_name, config=vars(args))

    # ── FM head (trainable) ───────────────────────────────────────────────────
    fm = FMHeadAVSR(
        n_layers=args.n_dit_layers,
        use_cross_attn=args.use_cross_attn,
    ).to(device).bfloat16()
    n_train = sum(p.numel() for p in fm.parameters())
    print(f"FM head: {n_train/1e6:.1f}M params (all trainable)")

    # ── Dataset ───────────────────────────────────────────────────────────────
    limit = 32 if args.debug else None
    if args.debug: args.batch_size = min(args.batch_size, 4)
    ds = FMAVSRDataset(args.data_root, args.split, subset="train",
                       limit=limit, clip_list=args.clip_list)
    val_n   = min(args.val_clips, len(ds))
    train_n = len(ds) - val_n
    if val_n > 0:
        train_ds, val_ds = random_split(ds, [train_n, val_n],
                                        generator=torch.Generator().manual_seed(42))
    else:
        train_ds, val_ds = ds, None
    print(f"train={train_n}  val={val_n}")

    steps_per_epoch = max(1, train_n // args.batch_size)
    max_steps    = steps_per_epoch * args.max_epochs
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
    step = 0; t_log = time.time(); t_start = time.time()
    last_save_loss = None
    last_save_recon = None
    last_save_sample_recon = None
    last_save_total = None
    last_save_lr = None
    fm.train()

    try:
        for epoch in range(100_000):
            for batch in train_loader:
                enc    = torch.from_numpy(batch["enc"]).to(device, dtype=torch.bfloat16)
                lat_gt = torch.from_numpy(batch["latent"]).to(device, dtype=torch.float32)
                spk    = torch.from_numpy(batch["speaker"]).to(device, dtype=torch.bfloat16)
                B, T_a = lat_gt.shape[:2]

                # Visual condition: downsample enc to T_a
                v_down = enc[:, ::2, :][:, :T_a, :]  # (B, T_a, 768)
                if v_down.shape[1] < T_a:
                    pad = torch.zeros(B, T_a - v_down.shape[1], 768,
                                      device=device, dtype=torch.bfloat16)
                    v_down = torch.cat([v_down, pad], 1)

                # Text condition: from pre-extracted smollm2_h.npy
                if args.no_text_cond or batch["h_lm"] is None:
                    h_down = torch.zeros(B, T_a, 960, device=device, dtype=torch.bfloat16)
                else:
                    h_down = resample_h_lm(batch["h_lm"], batch["lens_L"], T_a, device)

                # FM loss
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    loss_fm = fm.forward_train(v_down, h_down, spk, lat_gt)
                    if args.lambda_recon > 0:
                        pred_recon = fm.reconstruct_from_cond(v_down, h_down, spk)
                        loss_recon = torch.nn.functional.mse_loss(
                            pred_recon.float(), lat_gt.float()
                        )
                    else:
                        loss_recon = loss_fm.new_zeros(())
                    if args.lambda_sample_recon > 0:
                        pred_sample = fm.forward_inference(
                            v_down, h_down, spk, nfe=args.sample_recon_nfe
                        )
                        loss_sample_recon = torch.nn.functional.mse_loss(
                            pred_sample.float(), lat_gt.float()
                        )
                    else:
                        loss_sample_recon = loss_fm.new_zeros(())
                    loss = (
                        loss_fm
                        + args.lambda_recon * loss_recon
                        + args.lambda_sample_recon * loss_sample_recon
                    )

                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(fm.parameters(), 1.0)
                lr = get_lr(step, warmup_steps, max_steps, args.lr)
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
                    "loss_total": f"{loss.item():.8f}",
                    "lr": f"{lr:.10e}",
                    "seconds_per_step": "",
                    "elapsed_seconds": f"{elapsed:.4f}",
                })
                metrics_f.flush()
                last_save_loss = loss_fm.item()
                last_save_recon = loss_recon.item()
                last_save_sample_recon = loss_sample_recon.item()
                last_save_total = loss.item()
                last_save_lr = lr

                if step % 10 == 0:
                    sps = (time.time() - t_log) / 10; t_log = time.time()
                    eta = f"{(max_steps-step)*sps/3600:.1f}h"
                    print(f"step {step:6d}/{max_steps} | fm {loss_fm.item():.4f} | "
                          f"recon {loss_recon.item():.4f} | "
                          f"sample {loss_sample_recon.item():.4f} | "
                          f"total {loss.item():.4f} | "
                          f"lr {lr:.2e} | {sps:.2f}s/step | eta {eta}", flush=True)
                    if use_wandb:
                        wandb.log({
                            "train/loss_fm": loss_fm.item(),
                            "train/loss_recon": loss_recon.item(),
                            "train/loss_sample_recon": loss_sample_recon.item(),
                            "train/loss_total": loss.item(),
                            "train/lr": lr,
                        }, step=step)

                if val_loader is not None and step % args.eval_every == 0:
                    fm.eval()
                    total = 0.0; n = 0
                    with torch.no_grad():
                        for vb in val_loader:
                            enc_v  = torch.from_numpy(vb["enc"]).to(device, dtype=torch.bfloat16)
                            lat_v  = torch.from_numpy(vb["latent"]).to(device, dtype=torch.float32)
                            spk_v  = torch.from_numpy(vb["speaker"]).to(device, dtype=torch.bfloat16)
                            B_v    = lat_v.shape[0]; T_a_v = lat_v.shape[1]
                            vd_v   = enc_v[:, ::2, :][:, :T_a_v, :]
                            if vd_v.shape[1] < T_a_v:
                                pad = torch.zeros(B_v, T_a_v - vd_v.shape[1], 768,
                                                  device=device, dtype=torch.bfloat16)
                                vd_v = torch.cat([vd_v, pad], 1)
                            if args.no_text_cond or vb["h_lm"] is None:
                                hd_v = torch.zeros(B_v, T_a_v, 960, device=device, dtype=torch.bfloat16)
                            else:
                                hd_v = resample_h_lm(vb["h_lm"], vb["lens_L"], T_a_v, device)
                            total += fm.forward_train(vd_v, hd_v, spk_v, lat_v).item()
                            n += 1
                            if n >= 10: break
                    val_fm = total / max(n, 1)
                    print(f"  [val] loss_fm {val_fm:.4f}", flush=True)
                    val_metrics.writerow({
                        "step": step,
                        "epoch": epoch,
                        "val_loss_fm": f"{val_fm:.8f}",
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
