"""
StreamLip Offline Phase 1: lip-reading text prediction.

Standard next-token prediction on transcripts, conditioned on full-video
AV-HuBERT features via Flamingo-style cross-attention.

No SIL, no MFA alignment, no frame-level supervision.

Usage:
  python scripts/train_offline.py --debug --no_wandb
  python scripts/train_offline.py --lr 3e-4 --run_name offline_v1
"""
import argparse, math, os, sys, time
from pathlib import Path

_TMPDIR = Path(__file__).parent.parent / ".tmp"
_TMPDIR.mkdir(exist_ok=True)
os.environ["TMPDIR"] = str(_TMPDIR)

import torch, torch.multiprocessing
from torch.utils.data import DataLoader, random_split

torch.multiprocessing.set_sharing_strategy("file_system")
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from streaminlip.offline import StreamLipOffline
from streaminlip.offline.dataset import LRS3DatasetOffline, collate_fn
from transformers import AutoTokenizer

try:
    import wandb; _WANDB = True
except ImportError:
    _WANDB = False


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root",         default="data/processed")
    p.add_argument("--avhubert_ckpt",     default="pretrained/av-hubert/model.pt")
    p.add_argument("--gemma_path",        default="pretrained/gemma-3-1b")
    p.add_argument("--output_dir",        default="runs/offline/phase1")
    p.add_argument("--run_name",          default="offline_phase1")
    p.add_argument("--lora_rank",         type=int,   default=16)
    p.add_argument("--split",             default="pretrain")
    p.add_argument("--max_frames",        type=int,   default=150)
    p.add_argument("--max_text_len",      type=int,   default=64)
    p.add_argument("--batch_size",        type=int,   default=16)
    p.add_argument("--grad_accum",        type=int,   default=1)
    p.add_argument("--lr",                type=float, default=3e-4)
    p.add_argument("--warmup_epochs",     type=float, default=3.0)
    p.add_argument("--max_epochs",        type=int,   default=30)
    p.add_argument("--eval_every",        type=int,   default=500)
    p.add_argument("--save_every",        type=int,   default=1000)
    p.add_argument("--num_workers",       type=int,   default=4)
    p.add_argument("--val_clips",         type=int,   default=500)
    p.add_argument("--no_wandb",          action="store_true")
    p.add_argument("--debug",             action="store_true")
    return p.parse_args()


def get_lr(step, args):
    if step < args.warmup_steps:
        return args.lr * step / max(args.warmup_steps, 1)
    t = (step - args.warmup_steps) / max(args.max_steps - args.warmup_steps, 1)
    return args.lr * 0.5 * (1.0 + math.cos(math.pi * t))


def _worker_init(wid):
    import os as _os, torch.multiprocessing as _mp
    _os.environ["TMPDIR"] = str(Path(__file__).parent.parent / ".tmp")
    _mp.set_sharing_strategy("file_system")


def wer(ref: str, hyp: str) -> float:
    r, h = ref.upper().split(), hyp.upper().split()
    if not r: return 0.0
    d = list(range(len(h) + 1))
    for rw in r:
        d2 = [d[0] + 1]
        for j, hw in enumerate(h):
            d2.append(min(d[j+1]+1, d2[j]+1, d[j]+(0 if rw==hw else 1)))
        d = d2
    return d[-1] / len(r)


@torch.no_grad()
def evaluate(model, loader, tokenizer, device, max_batches=20):
    model.eval()
    total_loss, total_wer, n_wer, n = 0.0, 0.0, 0, 0
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        vis  = batch["visual"].to(device, dtype=torch.bfloat16)
        iids = batch["input_ids"].to(device)
        amsk = batch["attention_mask"].to(device)
        labs = batch["labels"].to(device)

        out = model(visual=vis, input_ids=iids, attention_mask=amsk, labels=labs)
        total_loss += out["loss"].item()
        n += 1

        # WER on first sample of each batch (generate is slow — sample only)
        ids = model.generate_text(vis[:1], max_new_tokens=64)
        pred = tokenizer.decode(ids[0], skip_special_tokens=True).strip()
        # recover GT from labels (ignore -100 padding)
        gt_ids = labs[0][labs[0] != -100].tolist()
        gt = tokenizer.decode(gt_ids, skip_special_tokens=True).strip()
        total_wer += wer(gt, pred)
        n_wer += 1

    return total_loss / max(n, 1), total_wer / max(n_wer, 1)


def save_checkpoint(model, out_dir, step):
    path = out_dir / f"step_{step:06d}.pt"
    save_dict = {
        "step":             step,
        "cross_attn_layers": model.cross_attn_layers.state_dict(),
    }
    # Save LoRA weights (lora_ keys only, much smaller than full model)
    lora_state = {k: v for k, v in model.lm.state_dict().items() if "lora_" in k}
    if lora_state:
        save_dict["lm_lora"] = lora_state
    torch.save(save_dict, str(path))
    print(f"  [save] {path}", flush=True)


def main():
    args   = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)

    use_wandb = _WANDB and not args.no_wandb
    if use_wandb:
        wandb.init(entity="gzh-thu", project="StreamLip",
                   name=args.run_name, config=vars(args))

    limit = 200 if args.debug else None
    if args.debug:
        args.batch_size = min(args.batch_size, 4)

    ds = LRS3DatasetOffline(
        args.data_root, args.split, args.gemma_path,
        max_frames=args.max_frames, max_text_len=args.max_text_len,
        subset="train", limit=limit,
    )
    val_n = min(args.val_clips, max(1, len(ds) // 10))
    train_ds, val_ds = random_split(
        ds, [len(ds) - val_n, val_n],
        generator=torch.Generator().manual_seed(42))
    print(f"train={len(train_ds)}  val={val_n}")

    steps_per_epoch  = max(1, len(train_ds) // (args.batch_size * args.grad_accum))
    args.max_steps   = steps_per_epoch * args.max_epochs
    args.warmup_steps = max(1, int(steps_per_epoch * args.warmup_epochs))
    print(f"  steps/epoch={steps_per_epoch} | warmup={args.warmup_steps} | max={args.max_steps}")

    loader_kw = dict(batch_size=args.batch_size, collate_fn=collate_fn,
                     pin_memory=False, drop_last=True,
                     num_workers=args.num_workers)
    if args.num_workers > 0:
        loader_kw.update(dict(multiprocessing_context="spawn",
                              worker_init_fn=_worker_init,
                              persistent_workers=True, prefetch_factor=2))
    train_loader = DataLoader(train_ds, shuffle=True, **loader_kw)
    val_loader   = DataLoader(val_ds, shuffle=False, batch_size=args.batch_size,
                              num_workers=0, collate_fn=collate_fn, pin_memory=False)

    print("Building StreamLipOffline...")
    model = StreamLipOffline(args.avhubert_ckpt, args.gemma_path,
                             lora_rank=args.lora_rank).to(device).bfloat16()
    tokenizer = AutoTokenizer.from_pretrained(args.gemma_path)
    counts = model.param_counts()
    print(f"  Total {counts['total_M']}M | Trainable {counts['trainable_M']}M")

    # Gate params get 100x lr to overcome bfloat16 precision floor (~0.0005 near 0.1)
    gate_params  = [p for n, p in model.named_parameters() if p.requires_grad and "gate" in n]
    other_params = [p for n, p in model.named_parameters() if p.requires_grad and "gate" not in n]
    optimizer = torch.optim.AdamW([
        {"params": gate_params,  "lr": args.lr * 100, "lr_mult": 100},
        {"params": other_params, "lr": args.lr,        "lr_mult": 1},
    ], betas=(0.9, 0.95), weight_decay=0.1)
    print(f"  gate params: {len(gate_params)}  other params: {len(other_params)}")

    step = micros = 0
    model.train(); optimizer.zero_grad()
    t_log = time.time()

    for _ in range(100_000):
        for batch in train_loader:
            out = model(
                visual=batch["visual"].to(device, dtype=torch.bfloat16),
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                labels=batch["labels"].to(device),
            )
            (out["loss"] / args.grad_accum).backward()
            micros += 1
            if micros % args.grad_accum != 0:
                continue

            all_trainable = gate_params + other_params
            torch.nn.utils.clip_grad_norm_(all_trainable, 1.0)
            base_lr = get_lr(step, args)
            for pg in optimizer.param_groups:
                pg["lr"] = base_lr * pg.get("lr_mult", 1)
            optimizer.step(); optimizer.zero_grad()
            step += 1

            if step % 10 == 0 or (args.debug and step <= 3):
                sps = (time.time() - t_log) / max(step % 10 or 10, 1)
                t_log = time.time()
                rem = (args.max_steps - step) * sps
                eta = f"{rem/3600:.1f}h" if rem > 3600 else f"{rem/60:.0f}m"
                print(f"step {step:6d}/{args.max_steps} | loss {out['loss'].item():.4f} | "
                      f"lr {get_lr(step,args):.2e} | {sps:.2f}s/step | eta {eta}", flush=True)
                if use_wandb:
                    wandb.log({"train/loss": out["loss"].item(),
                               "train/lr": get_lr(step, args)}, step=step)

            if step % args.eval_every == 0:
                val_loss, val_wer = evaluate(model, val_loader, tokenizer, device)
                print(f"  [val] loss {val_loss:.4f} | WER {val_wer:.1%}", flush=True)
                if use_wandb:
                    wandb.log({"val/loss": val_loss, "val/wer": val_wer}, step=step)
                model.train()

            if step % args.save_every == 0:
                save_checkpoint(model, out_dir, step)

            if args.debug and step >= 3:
                print("Debug run complete."); return
            if step >= args.max_steps:
                save_checkpoint(model, out_dir, step)
                print("Training complete."); return


if __name__ == "__main__":
    main()
