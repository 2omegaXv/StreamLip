"""
StreamLip V4 — Phase 2 training (FM head only).

Loads a Phase 1 checkpoint, freezes everything except FM head,
trains OT-CFM loss on Mimi latents.

Ablation flag --no_text_cond zeros out h_lm condition →
directly compares "with text prior" vs "visual-only" FM.

Usage:
  python scripts/train_v4_phase2.py \
      --phase1_ckpt runs/v4/v4_full_chunk_gate_fp32_lr3e-4_ep30/step_013290.pt \
      --run_name v4_p2_with_text

  python scripts/train_v4_phase2.py \
      --phase1_ckpt runs/v4/v4_full_chunk_gate_fp32_lr3e-4_ep30/step_013290.pt \
      --no_text_cond --run_name v4_p2_no_text
"""
import argparse
import math
import os
import sys
import time
from pathlib import Path

_TMPDIR = Path(__file__).parent.parent / ".tmp"
_TMPDIR.mkdir(exist_ok=True)
os.environ["TMPDIR"] = str(_TMPDIR)

import torch
import torch.multiprocessing
from torch.utils.data import DataLoader, random_split

torch.multiprocessing.set_sharing_strategy("file_system")
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from streaminlip.v4 import StreamLipV4
from streaminlip.v4.data.dataset import LRS3DatasetV4, collate_fn

try:
    import wandb
    _WANDB = True
except ImportError:
    _WANDB = False


def _worker_init(worker_id):
    import os
    os.environ["TMPDIR"] = str(Path(__file__).parent.parent / ".tmp")
    torch.multiprocessing.set_sharing_strategy("file_system")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--phase1_ckpt",      required=True)
    p.add_argument("--split",            default="pretrain")
    p.add_argument("--data_root",        default="data/processed")
    p.add_argument("--avhubert_ckpt",    default="pretrained/av-hubert/model.pt")
    p.add_argument("--smollm2_path",     default="pretrained/smollm2-360m")
    p.add_argument("--resnet50_weights", default="pretrained/resnet50-11ad3fa6.pth")
    p.add_argument("--output_dir",       default="runs/v4/phase2")
    p.add_argument("--run_name",         default="v4_p2_with_text")
    p.add_argument("--no_text_cond",     action="store_true",
                   help="Ablation: zero h_lm in FM condition (visual-only baseline)")
    p.add_argument("--batch_size",       type=int,   default=64)
    p.add_argument("--lr",               type=float, default=2e-4)
    p.add_argument("--warmup_epochs",    type=float, default=2.0)
    p.add_argument("--max_epochs",       type=int,   default=60)
    p.add_argument("--max_frames",       type=int,   default=150)
    p.add_argument("--eval_every",       type=int,   default=500)
    p.add_argument("--save_every",       type=int,   default=1000)
    p.add_argument("--num_workers",      type=int,   default=4)
    p.add_argument("--val_clips",        type=int,   default=500)
    p.add_argument("--lora_rank",        type=int,   default=16)
    p.add_argument("--no_wandb",         action="store_true")
    p.add_argument("--debug",            action="store_true")
    return p.parse_args()


def get_lr(step: int, warmup_steps: int, max_steps: int, lr: float) -> float:
    if step < warmup_steps:
        return lr * step / max(warmup_steps, 1)
    t = (step - warmup_steps) / max(max_steps - warmup_steps, 1)
    return lr * 0.5 * (1.0 + math.cos(math.pi * t))


@torch.no_grad()
def evaluate(model, loader, device, max_batches=20):
    model.eval()
    total_fm = 0.0
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
        latent       = batch["latent"].to(device, dtype=torch.float32) \
                       if batch["latent"] is not None else None
        if latent is None:
            continue
        out = model(visual=visual, clean_ids=clean_ids, clean_mask=clean_mask,
                    lm_idx_text=lm_idx_text, lm_idx_fm=lm_idx_fm,
                    face=face, frame_labels=frame_labels, mask=mask, latent=latent)
        total_fm += out["loss_fm"].item()
    return total_fm / max(i + 1, 1)


def save_checkpoint(model, out_dir, step):
    path = out_dir / f"step_{step:06d}.pt"
    torch.save({"step": step, "fm_head": model.fm_head.state_dict()}, str(path))
    print(f"  [save] {path}", flush=True)


def main():
    args   = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path(args.output_dir) / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    cond_label = "no_text" if args.no_text_cond else "with_text"
    print(f"Phase 2 | condition: {cond_label} | ckpt: {args.phase1_ckpt}")

    use_wandb = _WANDB and not args.no_wandb
    if use_wandb:
        wandb.init(entity="gzh-thu", project="StreamLip",
                   name=args.run_name, config=vars(args))

    # ── dataset ──────────────────────────────────────────────────────────────
    limit = 32 if args.debug else None
    if args.debug:
        args.batch_size = min(args.batch_size, 4)
    ds = LRS3DatasetV4(
        processed_root=args.data_root, split=args.split,
        tokenizer_path=args.smollm2_path, max_frames=args.max_frames,
        load_face=True, load_latent=True, subset="train", limit=limit,
    )
    val_n   = min(args.val_clips, max(1, len(ds) // 10))
    train_n = len(ds) - val_n
    train_ds, val_ds = random_split(ds, [train_n, val_n],
                                    generator=torch.Generator().manual_seed(42))
    print(f"train={train_n}  val={val_n}")

    steps_per_epoch = max(1, train_n // args.batch_size)
    max_steps       = steps_per_epoch * args.max_epochs
    warmup_steps    = max(1, int(steps_per_epoch * args.warmup_epochs))
    print(f"  steps/epoch={steps_per_epoch} | warmup={warmup_steps} | max={max_steps}")

    loader_kw = dict(batch_size=args.batch_size, collate_fn=collate_fn,
                     pin_memory=False, drop_last=True, num_workers=args.num_workers)
    if args.num_workers > 0:
        loader_kw.update(dict(multiprocessing_context="spawn",
                              worker_init_fn=_worker_init, persistent_workers=True,
                              prefetch_factor=2))
    train_loader = DataLoader(train_ds, shuffle=True,  **loader_kw)
    val_loader   = DataLoader(val_ds,   shuffle=False,
                              batch_size=args.batch_size, num_workers=0,
                              collate_fn=collate_fn, pin_memory=False)

    # ── model ────────────────────────────────────────────────────────────────
    print("Building model...")
    model = StreamLipV4(
        avhubert_ckpt=args.avhubert_ckpt, smollm2_path=args.smollm2_path,
        lora_rank=args.lora_rank, no_text_cond=args.no_text_cond,
        resnet50_weights=args.resnet50_weights,
    )

    # Load Phase 1 weights
    ckpt = torch.load(args.phase1_ckpt, map_location="cpu", weights_only=False)
    model.visual_encoder.load_state_dict(ckpt["visual_encoder"])
    model.sil_head.load_state_dict(ckpt["sil_head"])
    model.lm.load_state_dict(ckpt["lm"])
    print(f"  Loaded phase1: {args.phase1_ckpt}")

    model.phase2_mode()
    model = model.to(device).bfloat16()
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable (FM head): {n_train/1e6:.1f}M")

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, betas=(0.9, 0.95), weight_decay=0.1,
    )

    # ── training loop ────────────────────────────────────────────────────────
    step = 0
    model.train()
    optimizer.zero_grad()
    t_log = time.time()

    for _ in range(100_000):
        for batch in train_loader:
            latent = batch["latent"]
            if latent is None or latent.shape[1] == 0:
                continue

            visual       = batch["visual"].to(device, dtype=torch.bfloat16)
            clean_ids    = batch["clean_ids"].to(device)
            clean_mask   = batch["clean_mask"].to(device)
            lm_idx_text  = batch["lm_idx_text"].to(device)
            lm_idx_fm    = batch["lm_idx_fm"].to(device)
            frame_labels = batch["frame_labels"].to(device)
            mask         = batch["mask"].to(device)
            face         = batch["face"].to(device, dtype=torch.bfloat16)
            latent       = latent.to(device, dtype=torch.float32)

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                out = model(visual=visual, clean_ids=clean_ids, clean_mask=clean_mask,
                            lm_idx_text=lm_idx_text, lm_idx_fm=lm_idx_fm,
                            face=face, frame_labels=frame_labels, mask=mask,
                            latent=latent)

            out["loss_fm"].backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0)
            lr = get_lr(step, warmup_steps, max_steps, args.lr)
            for pg in optimizer.param_groups:
                pg["lr"] = lr
            optimizer.step()
            optimizer.zero_grad()
            step += 1

            if step % 10 == 0:
                now = time.time()
                sps = (now - t_log) / 10
                t_log = now
                rem = (max_steps - step) * sps
                eta = f"{rem/3600:.1f}h" if rem > 3600 else f"{rem/60:.0f}m"
                print(f"step {step:6d}/{max_steps} | fm {out['loss_fm'].item():.4f} | "
                      f"lr {lr:.2e} | {sps:.2f}s/step | eta {eta}", flush=True)
                if use_wandb:
                    wandb.log({"train/loss_fm": out["loss_fm"].item(),
                               "train/lr": lr}, step=step)

            if step % args.eval_every == 0:
                val_fm = evaluate(model, val_loader, device)
                print(f"  [val] loss_fm {val_fm:.4f}", flush=True)
                if use_wandb:
                    wandb.log({"val/loss_fm": val_fm}, step=step)
                model.train()

            if step % args.save_every == 0:
                save_checkpoint(model, out_dir, step)

            if args.debug and step >= 3:
                print("Debug done.")
                return

            if step >= max_steps:
                save_checkpoint(model, out_dir, step)
                print("Training complete.")
                return


if __name__ == "__main__":
    main()
