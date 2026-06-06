"""
StreamLip V4 — Phase 1 training (text path only, no FM).

Key differences from V3:
  - Dataset always returns raw lip frames; AV-HuBERT runs online.
  - T = random chunk-aligned prefix (k*C), eliminating train/inference mismatch.
  - No Conformer; AV-HuBERT mid-layer → cross-attn K/V.
  - AV-HuBERT LoRA (rank=16) replaces Conformer capacity.

Usage:
  python scripts/train_v4.py --debug --no_wandb
  python scripts/train_v4.py --run_name v4_exp1 --lr 3e-4
"""
import argparse
import math
import os
import sys
import time
from pathlib import Path

# Redirect shared-memory tensor storage off /dev/shm.
# torch.multiprocessing's "file_system" strategy writes temp files to TMPDIR.
_TMPDIR = Path(__file__).parent.parent / ".tmp"
_TMPDIR.mkdir(exist_ok=True)
os.environ["TMPDIR"] = str(_TMPDIR)

import torch
import torch.multiprocessing
from torch.utils.data import DataLoader, random_split

torch.multiprocessing.set_sharing_strategy("file_system")
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def _worker_init(worker_id):
    import os
    os.environ["TMPDIR"] = str(Path(__file__).parent.parent / ".tmp")
    torch.multiprocessing.set_sharing_strategy("file_system")


from streaminlip.v4 import StreamLipV4
from streaminlip.v4.data.dataset import LRS3DatasetV4, collate_fn

try:
    import wandb
    _WANDB = True
except ImportError:
    _WANDB = False


# ── args ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--split",               default="pretrain")
    p.add_argument("--data_root",           default="data/processed")
    p.add_argument("--avhubert_ckpt",       default="pretrained/self_large_vox_433h.pt")
    p.add_argument("--smollm2_path",        default="pretrained/smollm2-360m")
    p.add_argument("--resnet50_weights",    default="pretrained/resnet50-11ad3fa6.pth")
    p.add_argument("--output_dir",          default="runs/v4/phase1")
    p.add_argument("--run_name",            default="streamlip_v4_phase1")
    p.add_argument("--batch_size",          type=int,   default=256)
    p.add_argument("--grad_accum",          type=int,   default=1)
    p.add_argument("--lr",                  type=float, default=3e-4)
    p.add_argument("--warmup_steps",        type=int,   default=None)
    p.add_argument("--warmup_epochs",       type=float, default=3.0)
    p.add_argument("--max_steps",           type=int,   default=None)
    p.add_argument("--max_epochs",          type=int,   default=30)
    p.add_argument("--max_frames",          type=int,   default=150)
    p.add_argument("--cross_attn_every_n",  type=int,   default=4)
    p.add_argument("--lora_rank",           type=int,   default=16)
    p.add_argument("--lambda_text",         type=float, default=1.0)
    p.add_argument("--lambda_sil",          type=float, default=0.3)
    p.add_argument("--lambda_fm",           type=float, default=1.0)
    p.add_argument("--load_latent",         action="store_true",
                   help="Joint training: load Mimi latents and train FM head simultaneously")
    p.add_argument("--load_face",           action="store_true",
                   help="Load face.npz for speaker encoder (required when --load_latent)")
    p.add_argument("--no_text_cond",        action="store_true",
                   help="Ablation: zero out h_lm in FM condition (visual-only FM baseline)")
    p.add_argument("--eval_every",          type=int,   default=500)
    p.add_argument("--save_every",          type=int,   default=1000)
    p.add_argument("--num_workers",         type=int,   default=4)
    p.add_argument("--val_clips",           type=int,   default=500)
    p.add_argument("--no_wandb",            action="store_true")
    p.add_argument("--debug",               action="store_true")
    return p.parse_args()


# ── lr schedule ───────────────────────────────────────────────────────────────

def get_lr(step: int, args) -> float:
    if step < args.warmup_steps:
        return args.lr * step / max(args.warmup_steps, 1)
    t = (step - args.warmup_steps) / max(args.max_steps - args.warmup_steps, 1)
    return args.lr * 0.5 * (1.0 + math.cos(math.pi * t))


# ── eval ──────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model: StreamLipV4, loader: DataLoader, device: str,
             max_batches: int = 20, load_latent: bool = False) -> tuple:
    model.eval()
    total_loss_text = total_loss_sil = total_loss_fm = 0.0
    total_correct = total_valid = 0
    total_sil_correct = total_sil_valid = 0
    i = -1
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        visual       = batch["visual"].to(device, dtype=torch.bfloat16)
        clean_ids    = batch["clean_ids"].to(device)
        clean_mask   = batch["clean_mask"].to(device)
        lm_idx_text  = batch["lm_idx_text"].to(device)
        lm_idx_fm    = batch["lm_idx_fm"].to(device)
        frame_labels = batch["frame_labels"].to(device)
        mask         = batch["mask"].to(device)
        face         = batch["face"].to(device, dtype=torch.bfloat16)
        latent       = (batch["latent"].to(device, dtype=torch.float32)
                        if load_latent and batch["latent"] is not None
                           and batch["latent"].shape[1] > 0 else None)
        ctc_ids      = batch["ctc_ids"].to(device)
        ctc_lens     = batch["ctc_lens"].to(device)

        out = model(
            visual=visual, clean_ids=clean_ids, clean_mask=clean_mask,
            lm_idx_text=lm_idx_text, lm_idx_fm=lm_idx_fm,
            face=face, frame_labels=frame_labels, mask=mask, latent=latent,
            ctc_ids=ctc_ids, ctc_lens=ctc_lens,
        )
        total_loss_sil  += out["loss_sil"].item()
        total_loss_fm   += out["loss_fm"].item() if load_latent else 0.0

        preds   = out["posterior"].argmax(-1)
        # word_acc on last chunk only — matches inference target task
        T_actual   = mask.sum(dim=1, keepdim=True)
        position   = torch.arange(mask.shape[1], device=mask.device).unsqueeze(0)
        last_chunk = (position >= (T_actual - 6)) & mask
        non_sil = last_chunk & (frame_labels != 16)
        total_correct += (preds[non_sil] == frame_labels[non_sil]).sum().item()
        total_valid   += non_sil.sum().item()

        sil_target = (frame_labels == 16)
        sil_pred   = out["sil_logit"] > 0
        total_sil_correct += (sil_pred[mask] == sil_target[mask]).sum().item()
        total_sil_valid   += mask.sum().item()

    n = max(i + 1, 1)
    return (
        total_loss_text / n,
        total_correct / max(total_valid, 1),
        total_loss_sil / n,
        total_sil_correct / max(total_sil_valid, 1),
        total_loss_fm / n,
    )


# ── checkpoint ────────────────────────────────────────────────────────────────

def save_checkpoint(model: StreamLipV4, out_dir: Path, step: int, load_latent: bool = False):
    path = out_dir / f"step_{step:06d}.pt"
    state = {
        "step":           step,
        "visual_encoder": model.visual_encoder.state_dict(),
        "sil_head":       model.sil_head.state_dict(),
        "visual_head":    model.visual_head.state_dict(),
        "lm":             model.lm.state_dict(),
    }
    if load_latent:
        state["fm_head"]         = model.fm_head.state_dict()
        state["speaker_encoder"] = model.speaker_encoder.state_dict()
    torch.save(state, str(path))
    print(f"  [save] {path}", flush=True)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    args   = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    use_wandb = _WANDB and not args.no_wandb
    if use_wandb:
        wandb.init(
            entity="gzh-thu",
            project="StreamLip",
            name=args.run_name,
            config=vars(args),
        )

    # ── dataset ───────────────────────────────────────────────────────────────
    limit = 200 if args.debug else None
    if args.debug:
        args.batch_size = min(args.batch_size, 4)
    ds = LRS3DatasetV4(
        processed_root=args.data_root,
        split=args.split,
        tokenizer_path=args.smollm2_path,
        max_frames=args.max_frames,
        load_face=args.load_face or args.load_latent,
        load_latent=args.load_latent,
        subset="train",
        limit=limit,
    )
    val_n   = min(args.val_clips, max(1, len(ds) // 10))
    train_n = len(ds) - val_n
    train_ds, val_ds = random_split(ds, [train_n, val_n],
                                    generator=torch.Generator().manual_seed(42))
    print(f"train={train_n}  val={val_n}")

    steps_per_epoch = max(1, train_n // (args.batch_size * args.grad_accum))
    if args.max_steps is None:
        args.max_steps = steps_per_epoch * args.max_epochs
    if args.warmup_steps is None:
        args.warmup_steps = max(1, int(steps_per_epoch * args.warmup_epochs))
    print(
        f"  steps/epoch={steps_per_epoch} | "
        f"warmup={args.warmup_steps} | max={args.max_steps}"
    )

    # V4 runs AV-HuBERT online — num_workers > 0 causes issues with CUDA fork.
    # Use num_workers=0 or spawn context.  Default: spawn.
    loader_kw = dict(
        batch_size=args.batch_size,
        collate_fn=collate_fn,
        pin_memory=False,
        drop_last=True,
        num_workers=args.num_workers,
    )
    if args.num_workers > 0:
        loader_kw.update(dict(
            multiprocessing_context="spawn",
            worker_init_fn=_worker_init,
            persistent_workers=True,
            prefetch_factor=2,
        ))
    train_loader = DataLoader(train_ds, shuffle=True,  **loader_kw)
    val_loader   = DataLoader(val_ds, shuffle=False,
                              batch_size=args.batch_size,
                              num_workers=0,
                              collate_fn=collate_fn,
                              pin_memory=False,
                              drop_last=False)

    # ── model ─────────────────────────────────────────────────────────────────
    print("Building StreamLipV4...")
    model = StreamLipV4(
        avhubert_ckpt=args.avhubert_ckpt,
        smollm2_path=args.smollm2_path,
        cross_attn_every_n=args.cross_attn_every_n,
        lora_rank=args.lora_rank,
        lambda_text=args.lambda_text,
        lambda_sil=args.lambda_sil,
        no_text_cond=args.no_text_cond,
        resnet50_weights=args.resnet50_weights,
    )
    if args.load_latent:
        # Joint training: unfreeze FM head in addition to phase1 modules
        model.phase1_mode()
        for p in model.fm_head.parameters():
            p.requires_grad_(True)
        for p in model.speaker_encoder.parameters():
            p.requires_grad_(True)
        print(f"  Joint training: CE + FM losses, lambda_fm={args.lambda_fm}")
    else:
        model.phase1_mode()
    model = model.to(device).bfloat16()

    counts = model.param_counts()
    print(f"  Total {counts['total_M']}M | "
          f"Trainable {counts['trainable_M']}M | "
          f"Frozen {counts['frozen_M']}M")

    # ── optimizer ─────────────────────────────────────────────────────────────
    # Gates need a much higher LR: stored in bfloat16, the precision near 0.5
    # is ~0.004, but normal lr * grad updates are ~3e-4 → updates round to 0.
    # Keep gates in fp32 AND use 100x lr to ensure they actually move.
    gate_ids = set()
    for ca in model.lm.cross_attn_layers:
        ca.gate.data = ca.gate.data.float()   # keep gate in fp32
        gate_ids.add(id(ca.gate))
    gate_params  = [p for p in model.parameters() if p.requires_grad and id(p) in gate_ids]
    other_params = [p for p in model.parameters() if p.requires_grad and id(p) not in gate_ids]
    optimizer = torch.optim.AdamW(
        [
            {"params": other_params, "lr": args.lr},
            {"params": gate_params,  "lr": args.lr * 100},
        ],
        betas=(0.9, 0.95), weight_decay=0.1,
    )
    print(f"  optim: {len(other_params)} regular + {len(gate_params)} gate params (gate lr×100, fp32)")

    # ── training loop ─────────────────────────────────────────────────────────
    step   = 0
    micros = 0
    model.train()
    optimizer.zero_grad()
    t_log = time.time()

    for _ in range(100_000):
        for batch in train_loader:

            visual       = batch["visual"].to(device, dtype=torch.bfloat16)
            clean_ids    = batch["clean_ids"].to(device)
            clean_mask   = batch["clean_mask"].to(device)
            lm_idx_text  = batch["lm_idx_text"].to(device)
            lm_idx_fm    = batch["lm_idx_fm"].to(device)
            frame_labels = batch["frame_labels"].to(device)
            mask         = batch["mask"].to(device)
            face         = batch["face"].to(device, dtype=torch.bfloat16)
            latent       = (batch["latent"].to(device, dtype=torch.float32)
                            if args.load_latent and batch["latent"] is not None
                               and batch["latent"].shape[1] > 0 else None)
            ctc_ids      = batch["ctc_ids"].to(device)
            ctc_lens     = batch["ctc_lens"].to(device)

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                out = model(
                    visual=visual, clean_ids=clean_ids, clean_mask=clean_mask,
                    lm_idx_text=lm_idx_text, lm_idx_fm=lm_idx_fm,
                    face=face, frame_labels=frame_labels, mask=mask, latent=latent,
                    ctc_ids=ctc_ids, ctc_lens=ctc_lens,
                )

            (out["loss"] / args.grad_accum).backward()
            micros += 1

            if micros % args.grad_accum != 0:
                continue

            torch.nn.utils.clip_grad_norm_(other_params + gate_params, 1.0)
            base_lr = get_lr(step, args)
            optimizer.param_groups[0]["lr"] = base_lr
            optimizer.param_groups[1]["lr"] = base_lr * 100
            optimizer.step()
            optimizer.zero_grad()
            step += 1

            if step % 10 == 0:
                now = time.time()
                sps = (now - t_log) / 10
                t_log = now
                rem = (args.max_steps - step) * sps
                eta = f"{rem/3600:.1f}h" if rem > 3600 else f"{rem/60:.0f}m"
                print(
                    f"step {step:6d}/{args.max_steps} | "
                    f"loss {out['loss'].item():.4f} | "
                    f"text {out['loss_text'].item():.4f} | "
                    f"sil {out['loss_sil'].item():.4f} | "
                    f"lr {get_lr(step, args):.2e} | {sps:.2f}s/step | eta {eta}",
                    flush=True,
                )
                if use_wandb:
                    wandb.log({
                        "train/loss":      out["loss"].item(),
                        "train/loss_text": out["loss_text"].item(),
                        "train/loss_sil":  out["loss_sil"].item(),
                        "train/loss_vis":  out["loss_visual"].item(),
                        **( {"train/loss_fm": out["loss_fm"].item()}
                            if args.load_latent else {} ),
                        "train/lr":        get_lr(step, args),
                        "perf/sps":        sps,
                    }, step=step)

            if step % args.eval_every == 0:
                val_loss_text, val_acc, val_loss_sil, val_sil_acc, val_loss_fm = evaluate(
                    model, val_loader, device, load_latent=args.load_latent)
                print(
                    f"  [val] loss_text {val_loss_text:.4f} | word_acc {val_acc:.4f} | "
                    f"loss_sil {val_loss_sil:.4f} | sil_acc {val_sil_acc:.4f}"
                    + (f" | loss_fm {val_loss_fm:.4f}" if args.load_latent else ""),
                    flush=True,
                )
                if use_wandb:
                    wandb.log({
                        "val/loss_text": val_loss_text, "val/word_acc": val_acc,
                        "val/loss_sil":  val_loss_sil,  "val/sil_acc":  val_sil_acc,
                        **( {"val/loss_fm": val_loss_fm} if args.load_latent else {} ),
                    }, step=step)
                model.train()

            if step % args.save_every == 0:
                save_checkpoint(model, out_dir, step, args.load_latent)

            if step % 1 == 0 and args.debug:   # log every step in debug
                print(
                    f"step {step:4d} | loss {out['loss'].item():.4f} | "
                    f"text {out['loss_text'].item():.4f} | sil {out['loss_sil'].item():.4f} | vis {out['loss_visual'].item():.4f}"
                    + (f" | fm {out['loss_fm'].item():.4f}" if args.load_latent else ""),
                    flush=True,
                )
            if args.debug and step >= 3:
                print("Debug run complete.")
                return

            if step >= args.max_steps:
                save_checkpoint(model, out_dir, step, args.load_latent)
                print("Training complete.")
                return


if __name__ == "__main__":
    main()
