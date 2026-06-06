"""
StreamLip Phase 1 training: visual cross-attention + CE loss on text.

Usage:
  cd /mnt/pfs/group-jt/zihan.guo/droid/DL-V2A
  .venv/bin/python scripts/train.py

Key flags:
  --split         pretrain | trainval  (default: pretrain)
  --batch_size    per-GPU batch size   (default: 4)
  --max_steps     training steps       (default: 100000)
  --eval_every    steps between evals  (default: 1000)
  --save_every    steps between ckpts  (default: 5000)
  --avhubert_ckpt path to av-hubert checkpoint
  --backbone_ckpt path to gemma-3-1b dir
  --run_name      WandB run name
"""
import argparse
import math
import os
import sys
import time
from pathlib import Path

import torch
import torch.multiprocessing
import torch.nn.functional as F
import wandb
from torch.utils.data import DataLoader, random_split

# Use file_system strategy instead of file_descriptor so workers don't hit /dev/shm limit
torch.multiprocessing.set_sharing_strategy("file_system")

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.streaminlip.model import StreamLip
from src.streaminlip.av_hubert import AVHuBERTExtractor
from src.streaminlip.data.dataset import LRS3Dataset, collate_fn


# ── args ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--split",        default="pretrain")
    p.add_argument("--data_root",    default="data/processed")
    p.add_argument("--avhubert_ckpt",default="pretrained/av-hubert/model.pt")
    p.add_argument("--backbone_ckpt",default="pretrained/gemma-3-1b")
    p.add_argument("--tokenizer",    default="pretrained/gemma-3-1b")
    p.add_argument("--output_dir",   default="runs/phase1")
    p.add_argument("--run_name",     default="streaminlip_phase1")
    p.add_argument("--batch_size",   type=int, default=512)
    p.add_argument("--grad_accum",   type=int, default=4)   # effective bs = batch_size * grad_accum
    p.add_argument("--lr",           type=float, default=3e-3)
    p.add_argument("--warmup_epochs", type=float, default=10.0,
                   help="warmup duration in epochs (converted to steps after dataset is loaded)")
    p.add_argument("--max_steps",    type=int, default=None,
                   help="optimizer steps; if None, computed from --max_epochs")
    p.add_argument("--max_epochs",   type=int, default=100)
    p.add_argument("--max_frames",   type=int, default=250)
    p.add_argument("--max_tokens",   type=int, default=128)
    p.add_argument("--eval_every",   type=int, default=100)
    p.add_argument("--save_every",   type=int, default=100)
    p.add_argument("--num_workers",  type=int, default=16)
    p.add_argument("--no_wandb",     action="store_true")
    p.add_argument("--debug",        action="store_true")  # 1 batch only
    return p.parse_args()


# ── lr schedule ───────────────────────────────────────────────────────────────

def get_lr(step: int, args) -> float:
    if step < args.warmup_steps:
        return args.lr * step / args.warmup_steps
    progress = (step - args.warmup_steps) / max(1, args.max_steps - args.warmup_steps)
    return args.lr * 0.5 * (1 + math.cos(math.pi * progress))


# ── training ──────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── WandB ─────────────────────────────────────────────────────────────
    if not args.no_wandb:
        wandb.init(
            entity="gzh-thu",
            project="StreamLip",
            name=args.run_name,
            config=vars(args),
        )

    # ── data ──────────────────────────────────────────────────────────────
    limit = 20 if args.debug else None
    dataset = LRS3Dataset(
        processed_root=args.data_root,
        split=args.split,
        tokenizer_path=args.tokenizer,
        max_frames=args.max_frames,
        max_text_tokens=args.max_tokens,
        use_cached_avhubert=True,
        limit=limit,
    )

    val_size = min(200, len(dataset) // 10)
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    # Compute steps from epochs
    steps_per_epoch = train_size // args.batch_size // args.grad_accum
    if args.max_steps is None:
        args.max_steps = steps_per_epoch * args.max_epochs
    args.warmup_steps = max(1, int(steps_per_epoch * args.warmup_epochs))
    print(
        f"  steps/epoch: {steps_per_epoch}  |  "
        f"warmup: {args.warmup_steps} steps ({args.warmup_epochs} epoch)  |  "
        f"max_steps: {args.max_steps} ({args.max_steps / steps_per_epoch:.1f} epochs)"
    )

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, collate_fn=collate_fn,
        pin_memory=False, drop_last=True,
        prefetch_factor=1 if args.num_workers > 0 else None,
        persistent_workers=args.num_workers > 0,
        multiprocessing_context="fork" if args.num_workers > 0 else None,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=collate_fn,
        pin_memory=False,
        prefetch_factor=1 if args.num_workers > 0 else None,
        persistent_workers=args.num_workers > 0,
        multiprocessing_context="fork" if args.num_workers > 0 else None,
    )

    # ── model ─────────────────────────────────────────────────────────────
    print("Loading StreamLip model...")
    model = StreamLip(
        avhubert_ckpt=None,          # features pre-extracted; no AV-HuBERT at train time
        pretrained_backbone=args.backbone_ckpt,
        random_init=False,
        cross_attn_every_n=2,
    ).to(device).bfloat16()

    counts = model.param_counts()
    print(f"  Total:     {counts['total_M']:.1f}M")
    print(f"  Trainable: {counts['trainable_M']:.1f}M  (MLP adapter + cross-attn)")
    print(f"  Frozen:    {counts['frozen_M']:.1f}M  (Gemma base)")

    # ── optimizer ─────────────────────────────────────────────────────────
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, betas=(0.9, 0.95), weight_decay=0.1)

    # ── training loop ──────────────────────────────────────────────────────
    step = 0
    model.train()
    optimizer.zero_grad()
    t_start = time.time()
    t_step  = time.time()

    for epoch in range(args.max_epochs if args.max_steps is None else 10000):
        for batch in train_loader:

            # Move to device
            av_feats   = batch["av_feats"].to(device, dtype=torch.bfloat16)
            input_ids  = batch["input_ids"].to(device)
            labels     = batch["labels"].to(device)
            attn_mask  = batch["attention_mask"].to(device)

            # Forward (model handles AV-HuBERT internally for raw frames)
            logits = model(av_feats, input_ids, attn_mask)  # (B, T, V)

            # CE loss
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=-100,
            )
            loss = loss / args.grad_accum
            loss.backward()

            if (step + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                lr = get_lr(step // args.grad_accum, args)
                for pg in optimizer.param_groups:
                    pg["lr"] = lr
                optimizer.step()
                optimizer.zero_grad()

            step += 1
            actual_step = step // args.grad_accum

            # ── logging ───────────────────────────────────────────────────
            log_every = 1 if args.debug else 10
            if step % log_every == 0:
                loss_val = loss.item() * args.grad_accum
                ppl = math.exp(min(loss_val, 20))
                now = time.time()
                secs_per_step = (now - t_step) / log_every
                remaining = (args.max_steps - actual_step) * secs_per_step * args.grad_accum
                eta = f"{remaining/3600:.1f}h" if remaining > 3600 else f"{remaining/60:.0f}m"
                elapsed = now - t_start
                t_step = now
                print(
                    f"step {actual_step:6d}/{args.max_steps} | "
                    f"loss {loss_val:.4f} | ppl {ppl:.1f} | "
                    f"lr {get_lr(actual_step, args):.2e} | "
                    f"{secs_per_step:.2f}s/step | eta {eta}",
                    flush=True,
                )
                if not args.no_wandb:
                    wandb.log({"train/loss": loss_val, "train/ppl": ppl,
                               "train/lr": get_lr(actual_step, args),
                               "perf/secs_per_step": secs_per_step}, step=actual_step)

            # ── eval ──────────────────────────────────────────────────────
            if not args.debug and actual_step > 0 and actual_step % args.eval_every == 0:
                val_loss = evaluate(model, val_loader, device, args)
                print(f"  [eval] loss {val_loss:.4f} | ppl {math.exp(min(val_loss, 20)):.1f}")
                if not args.no_wandb:
                    wandb.log({"val/loss": val_loss, "val/ppl": math.exp(min(val_loss, 20))},
                              step=actual_step)
                model.train()

            # ── checkpoint ────────────────────────────────────────────────
            if not args.debug and actual_step > 0 and actual_step % args.save_every == 0:
                _save_checkpoint(model, out_dir, actual_step)

            if args.debug and step >= 3:
                print("Training complete (debug).", flush=True)
                return

            if actual_step >= args.max_steps:
                _save_checkpoint(model, out_dir, actual_step)
                print("Training complete.")
                return


def _save_checkpoint(model, out_dir: Path, step: int):
    save_path = out_dir / f"step_{step:06d}.pt"
    torch.save({
        "step": step,
        "visual_encoder": model.visual_encoder.state_dict(),
        "cross_attn_layers": model.cross_attn_layers.state_dict(),
        "av_hubert_stem": (model.av_hubert.model
                           .feature_extractor_video.resnet.frontend3D[0].weight.data
                           if model.av_hubert else None),
    }, str(save_path))
    print(f"  [save] {save_path}", flush=True)


@torch.no_grad()
def evaluate(model, loader, device, args) -> float:
    model.eval()
    total_loss, total_n = 0.0, 0
    for i, batch in enumerate(loader):
        if i >= 20:
            break
        av_feats  = batch["av_feats"].to(device, dtype=torch.bfloat16)
        input_ids = batch["input_ids"].to(device)
        labels    = batch["labels"].to(device)
        attn_mask = batch["attention_mask"].to(device)
        logits = model(av_feats, input_ids, attn_mask)
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
            ignore_index=-100,
        )
        total_loss += loss.item()
        total_n += 1
    return total_loss / max(total_n, 1)


if __name__ == "__main__":
    main()
