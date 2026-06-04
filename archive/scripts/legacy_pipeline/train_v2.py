"""
StreamLip V2 — Phase 1 training (text path only, no FM).

Usage:
  python scripts/train_v2.py
  python scripts/train_v2.py --debug
  python scripts/train_v2.py --run_name exp1 --batch_size 4 --lora_rank 0

Phase 1: VisualEncoder (AV-HuBERT stem + Conformer + VisualHead) and LMBackbone
are trained via CE loss on PoE posterior logits. FM Head and Speaker Encoder frozen.
"""
import argparse
import math
import sys
import time
from pathlib import Path

import torch
import torch.multiprocessing
from torch.utils.data import DataLoader, random_split

torch.multiprocessing.set_sharing_strategy("file_system")
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from streaminlip.v2 import StreamLipV2
from streaminlip.v2.data.dataset import LRS3DatasetV2, collate_fn

try:
    import wandb
    _WANDB = True
except ImportError:
    _WANDB = False


# ── args ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--split",            default="pretrain")
    p.add_argument("--data_root",        default="data/processed")
    p.add_argument("--avhubert_ckpt",    default="pretrained/av-hubert/model.pt")
    p.add_argument("--smollm2_path",     default="pretrained/smollm2-360m")
    p.add_argument("--resnet50_weights", default="pretrained/resnet50-11ad3fa6.pth")
    p.add_argument("--output_dir",       default="runs/v2/phase1")
    p.add_argument("--run_name",         default="streamlip_v2_phase1")
    p.add_argument("--batch_size",       type=int,   default=512)
    p.add_argument("--grad_accum",       type=int,   default=1)    # effective bs=32
    p.add_argument("--lr",               type=float, default=1e-3)
    p.add_argument("--warmup_steps",     type=int,   default=None,
                   help="warmup steps; overrides --warmup_epochs if set")
    p.add_argument("--warmup_epochs",    type=float, default=3.0,
                   help="warmup duration in epochs (converted to steps after dataset load)")
    p.add_argument("--max_steps",        type=int,   default=None,
                   help="total optimizer steps; overrides --max_epochs if set")
    p.add_argument("--max_epochs",       type=int,   default=25,
                   help="total epochs; used when --max_steps is not set")
    p.add_argument("--max_frames",       type=int,   default=150)
    p.add_argument("--lora_rank",        type=int,   default=16)
    p.add_argument("--alpha",            type=float, default=1.0,
                   help="PoE LM weight: posterior = log_softmax(s_vis) + alpha * log_softmax(s_lm)")
    p.add_argument("--lambda_text",      type=float, default=1.0,
                   help="text loss weight; use 1.0 for phase1, 0.005 for phase2 (FM dominates)")
    p.add_argument("--lambda_sil",       type=float, default=0.3,
                   help="SIL binary detection loss weight")
    p.add_argument("--eval_every",       type=int,   default=500)
    p.add_argument("--save_every",       type=int,   default=1000)
    p.add_argument("--num_workers",      type=int,   default=8)
    p.add_argument("--val_clips",        type=int,   default=500)
    p.add_argument("--no_wandb",         action="store_true")
    p.add_argument("--debug",            action="store_true")   # 3 steps then exit
    return p.parse_args()


# ── lr schedule ───────────────────────────────────────────────────────────────

def get_lr(step: int, args) -> float:
    if step < args.warmup_steps:
        return args.lr * step / max(args.warmup_steps, 1)
    t = (step - args.warmup_steps) / max(args.max_steps - args.warmup_steps, 1)
    return args.lr * 0.5 * (1.0 + math.cos(math.pi * t))


# ── eval ──────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model: StreamLipV2, loader: DataLoader, device: str,
             max_batches: int = 20) -> tuple[float, float, float, float]:
    model.eval()
    total_loss_text, total_loss_sil = 0.0, 0.0
    total_correct_nonsil, total_valid_nonsil = 0, 0
    total_sil_correct, total_sil_valid = 0, 0
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

        out = model(
            visual=visual, clean_ids=clean_ids, clean_mask=clean_mask,
            lm_idx_text=lm_idx_text, lm_idx_fm=lm_idx_fm,
            face=face, frame_labels=frame_labels, mask=mask, latent=None,
        )

        total_loss_text += out["loss_text"].item()
        total_loss_sil  += out["loss_sil"].item()

        preds   = out["posterior"].argmax(-1)
        non_sil = mask & (frame_labels != 16)
        total_correct_nonsil += (preds[non_sil] == frame_labels[non_sil]).sum().item()
        total_valid_nonsil   += non_sil.sum().item()

        sil_target = (frame_labels == 16)
        sil_pred   = out["sil_logit"] > 0
        total_sil_correct += (sil_pred[mask] == sil_target[mask]).sum().item()
        total_sil_valid   += mask.sum().item()

    n = max(i + 1, 1)
    return (
        total_loss_text / n,
        total_correct_nonsil / max(total_valid_nonsil, 1),
        total_loss_sil / n,
        total_sil_correct / max(total_sil_valid, 1),
    )


# ── checkpoint ────────────────────────────────────────────────────────────────

def save_checkpoint(model: StreamLipV2, out_dir: Path, step: int):
    path = out_dir / f"step_{step:06d}.pt"
    torch.save({
        "step":            step,
        "alpha":           model.alpha.data,
        "visual_encoder":  model.visual_encoder.state_dict(),
        "sil_head":        model.sil_head.state_dict(),
        "lm":              model.lm.state_dict(),
    }, str(path))
    print(f"  [save] {path}", flush=True)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    args    = parse_args()
    device  = "cuda" if torch.cuda.is_available() else "cpu"
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
    ds = LRS3DatasetV2(
        processed_root=args.data_root,
        split=args.split,
        tokenizer_path=args.smollm2_path,
        max_frames=args.max_frames,
        load_face=False,
        load_latent=False,
        subset="train",      # 保留最后 2000 条给 pretrain-test
        limit=limit,
    )
    val_n   = min(args.val_clips, len(ds) // 10)
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
        f"warmup={args.warmup_steps} ({args.warmup_steps / steps_per_epoch:.1f} ep) | "
        f"max={args.max_steps} ({args.max_steps / steps_per_epoch:.1f} ep)"
    )

    loader_kw = dict(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True,
        prefetch_factor=2 if args.num_workers > 0 else None,
        persistent_workers=args.num_workers > 0,
        multiprocessing_context="fork" if args.num_workers > 0 else None,
    )
    train_loader = DataLoader(train_ds, shuffle=True, **loader_kw)
    val_loader   = DataLoader(val_ds, shuffle=False,
                              batch_size=args.batch_size,
                              num_workers=min(args.num_workers, 4),
                              collate_fn=collate_fn,
                              pin_memory=True,
                              drop_last=False)

    # ── model ─────────────────────────────────────────────────────────────────
    print("Building StreamLipV2...")
    model = StreamLipV2(
        avhubert_ckpt=args.avhubert_ckpt,
        smollm2_path=args.smollm2_path,
        lora_rank=args.lora_rank,
        alpha=args.alpha,
        lambda_text=args.lambda_text,
        lambda_sil=args.lambda_sil,
        resnet50_weights=args.resnet50_weights,
    )
    model.phase1_mode()          # freeze FM head + speaker encoder
    model = model.to(device).bfloat16()

    counts = model.param_counts()
    print(f"  Total {counts['total_M']}M | "
          f"Trainable {counts['trainable_M']}M | "
          f"Frozen {counts['frozen_M']}M")

    # ── optimizer ─────────────────────────────────────────────────────────────
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable, lr=args.lr, betas=(0.9, 0.95), weight_decay=0.1
    )

    # ── training loop ─────────────────────────────────────────────────────────
    step   = 0       # optimizer steps
    micros = 0       # gradient-accumulation micro-steps
    model.train()
    optimizer.zero_grad()
    t_log = time.time()

    for _ in range(100_000):          # epoch loop
        for batch in train_loader:

            visual       = batch["visual"].to(device, dtype=torch.bfloat16)
            clean_ids    = batch["clean_ids"].to(device)
            clean_mask   = batch["clean_mask"].to(device)
            lm_idx_text  = batch["lm_idx_text"].to(device)
            lm_idx_fm    = batch["lm_idx_fm"].to(device)
            frame_labels = batch["frame_labels"].to(device)
            mask         = batch["mask"].to(device)
            face         = batch["face"].to(device, dtype=torch.bfloat16)

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                out = model(
                    visual=visual, clean_ids=clean_ids, clean_mask=clean_mask,
                    lm_idx_text=lm_idx_text, lm_idx_fm=lm_idx_fm,
                    face=face, frame_labels=frame_labels, mask=mask, latent=None,
                )

            (out["loss"] / args.grad_accum).backward()
            micros += 1

            if micros % args.grad_accum != 0:
                continue

            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            for pg in optimizer.param_groups:
                pg["lr"] = get_lr(step, args)
            optimizer.step()
            optimizer.zero_grad()
            step += 1

            # ── logging ───────────────────────────────────────────────────────
            if step % 10 == 0:
                loss_val  = out["loss"].item()
                loss_text = out["loss_text"].item()
                loss_sil  = out["loss_sil"].item()
                now       = time.time()
                sps       = (now - t_log) / 10
                t_log     = now
                rem = (args.max_steps - step) * sps
                eta = f"{rem/3600:.1f}h" if rem > 3600 else f"{rem/60:.0f}m"
                print(
                    f"step {step:6d}/{args.max_steps} | "
                    f"loss {loss_val:.4f} | text {loss_text:.4f} | sil {loss_sil:.4f} | "
                    f"lr {get_lr(step, args):.2e} | {sps:.2f}s/step | eta {eta}",
                    flush=True,
                )
                if use_wandb:
                    wandb.log({
                        "train/loss":      loss_val,
                        "train/loss_text": loss_text,
                        "train/loss_sil":  loss_sil,
                        "train/lr":        get_lr(step, args),
                        "perf/sps":        sps,
                    }, step=step)

            # ── eval ──────────────────────────────────────────────────────────
            if step % args.eval_every == 0:
                val_loss_text, val_acc_word, val_loss_sil, val_acc_sil = evaluate(model, val_loader, device)
                print(
                    f"  [val] loss_text {val_loss_text:.4f} | word_acc {val_acc_word:.4f} | "
                    f"loss_sil {val_loss_sil:.4f} | sil_acc {val_acc_sil:.4f}",
                    flush=True,
                )
                if use_wandb:
                    wandb.log({
                        "val/loss_text": val_loss_text, "val/word_acc": val_acc_word,
                        "val/loss_sil":  val_loss_sil,  "val/sil_acc":  val_acc_sil,
                    }, step=step)
                model.train()

            # ── checkpoint ────────────────────────────────────────────────────
            if step % args.save_every == 0:
                save_checkpoint(model, out_dir, step)

            if args.debug and step >= 3:
                print("Debug run complete.")
                return

            if step >= args.max_steps:
                save_checkpoint(model, out_dir, step)
                print("Training complete.")
                return


if __name__ == "__main__":
    main()
