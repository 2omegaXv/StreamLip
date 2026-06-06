"""
V5 训练脚本（对标 Auto-AVSR training）。

与 Auto-AVSR lightning.py 对齐：
  - decoder: SmolLM2 cross-attention 到 encoder 输出（替换小 Transformer decoder）
  - loss:    att_CE（全段文本，无 last_chunk_mask）+ λ_sil × SIL BCE
  - CTC:     encoder frozen，省略 CTC loss
  - optim:   AdamW + WarmupCosine（与 Auto-AVSR 一致）
  - data:    avsr_enc.npy (T,768) 预提取特征

Usage:
  python scripts/train_v5_avsr.py --run_name v5_ca_avsr
  python scripts/train_v5_avsr.py --run_name v5_ca_avsr --cross_attn_every_n 4
"""
import argparse, math, os, sys, time
from pathlib import Path

_TMPDIR = Path(__file__).parent.parent / ".tmp"
_TMPDIR.mkdir(exist_ok=True)
os.environ["TMPDIR"] = str(_TMPDIR)

import torch
import torch.multiprocessing
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split, Sampler

torch.multiprocessing.set_sharing_strategy("file_system")
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def _worker_init(_):
    os.environ["TMPDIR"] = str(Path(__file__).parent.parent / ".tmp")
    torch.multiprocessing.set_sharing_strategy("file_system")

from streaminlip.v5 import StreamLipV5
from streaminlip.v5.data import LRS3DatasetV5, collate_fn
from transformers import AutoTokenizer

try:
    import wandb; _WANDB = True
except ImportError:
    _WANDB = False


# ── Args ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run_name",         default="v5_ca_avsr")
    p.add_argument("--data_root",        default="data/processed")
    p.add_argument("--avsr_ckpt",        default="pretrained/auto_avsr/vsr_trlrs2lrs3vox2avsp_base.pth")
    p.add_argument("--smollm2_path",     default="pretrained/olmo-1b-lrs3-lr3e-5_ep2")
    p.add_argument("--output_dir",       default="runs/v5")
    p.add_argument("--cross_attn_every_n", type=int, default=4,
                   help="SmolLM2 每 N 层插一个 CA 层（对标 Auto-AVSR cross-attention decoder）")
    p.add_argument("--lora_rank",        type=int,   default=0)
    p.add_argument("--batch_size",       type=int,   default=256)
    p.add_argument("--grad_accum",       type=int,   default=1)
    p.add_argument("--lr",               type=float, default=3e-5)
    p.add_argument("--weight_decay",     type=float, default=0.01)
    p.add_argument("--warmup_epochs",    type=float, default=3.0)
    p.add_argument("--max_epochs",       type=int,   default=50)
    p.add_argument("--max_frames",       type=int,   default=150)
    p.add_argument("--eval_every",       type=int,   default=500)
    p.add_argument("--save_every",       type=int,   default=1000)
    p.add_argument("--val_clips",        type=int,   default=500)
    p.add_argument("--num_workers",      type=int,   default=16)
    p.add_argument("--limit",            type=int,   default=None)
    p.add_argument("--no_wandb",         action="store_true")
    return p.parse_args()


# ── WarmupCosine LR（对标 Auto-AVSR cosine.py） ────────────────────────────────

def get_lr(step: int, warmup: int, total: int, base_lr: float) -> float:
    if step < warmup:
        return base_lr * step / max(warmup, 1)
    t = (step - warmup) / max(total - warmup, 1)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * t))


# ── Bucketed sampler ───────────────────────────────────────────────────────────

class BucketBatchSampler(Sampler):
    def __init__(self, lengths, batch_size, shuffle=True):
        self.lengths    = lengths
        self.batch_size = batch_size
        self.shuffle    = shuffle

    def __iter__(self):
        import random
        idx = sorted(range(len(self.lengths)), key=lambda i: self.lengths[i], reverse=True)
        batches = [idx[i:i+self.batch_size] for i in range(0, len(idx), self.batch_size)]
        if batches and len(batches[-1]) < self.batch_size:
            batches = batches[:-1]
        if self.shuffle:
            random.shuffle(batches)
        for b in batches:
            yield b

    def __len__(self):
        return len(self.lengths) // self.batch_size


# ── Validation ────────────────────────────────────────────────────────────────

@torch.no_grad()
def run_val(model, val_loader, device, tokenizer, n_show=3):
    model.eval()
    tot_ce = tot_n = tot_correct = tot_valid = 0
    examples = []

    for batch in val_loader:
        if batch is None:
            continue
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            out = model(
                visual=batch["visual"],
                input_ids=batch["input_ids"],
                target_ids=batch["target_ids"],
                video_pos=batch["video_pos"],
                text_pos=batch["text_pos"],
                video_mask=batch["video_mask"],
                text_mask=batch["text_mask"],
                last_chunk_mask=None,
            )

        logits = out.get("text_logits")
        if logits is not None:
            valid = batch["target_ids"] != -100
            preds = logits.argmax(-1)
            tot_correct += (preds[valid] == batch["target_ids"][valid]).sum().item()
            tot_valid   += valid.sum().item()
            if len(examples) < n_show:
                for b in range(min(n_show - len(examples), batch["target_ids"].shape[0])):
                    v = batch["target_ids"][b] != -100
                    gt   = tokenizer.decode(batch["target_ids"][b][v].tolist(), skip_special_tokens=True)
                    pred = tokenizer.decode(logits[b].argmax(-1)[v].tolist(), skip_special_tokens=True)
                    examples.append((pred.strip(), gt.strip()))

        tot_ce += out["loss_ce"].item()
        tot_n  += 1

    model.train()
    if tot_n == 0:
        return {}, []
    return {
        "val_ce":      tot_ce  / tot_n,
        "val_tok_acc": tot_correct / max(tot_valid, 1),
    }, examples


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args   = parse_args()
    sys.stdout.reconfigure(line_buffering=True)
    device  = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path(args.output_dir) / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    use_wandb = _WANDB and not args.no_wandb
    if use_wandb:
        wandb.init(entity="gzh-thu", project="StreamLip", name=args.run_name)

    # ── Model ─────────────────────────────────────────────────────────────────
    model = StreamLipV5(
        avsr_ckpt=args.avsr_ckpt,
        smollm2_path=args.smollm2_path,
        lora_rank=args.lora_rank,
        cross_attn_every_n=args.cross_attn_every_n,
        cfg_drop_prob=0.0,
    ).to(device).bfloat16()

    counts = model.param_counts()
    print(f"[StreamLipV5] total={counts['total_M']}M  trainable={counts['trainable_M']}M  frozen={counts['frozen_M']}M")

    tokenizer = AutoTokenizer.from_pretrained(
        args.smollm2_path, clean_up_tokenization_spaces=False
    )

    # ── Data ──────────────────────────────────────────────────────────────────
    ds_full = LRS3DatasetV5(args.data_root, "pretrain", args.smollm2_path,
                            max_frames=args.max_frames, deterministic=False,
                            limit=args.limit)
    val_n   = min(args.val_clips, max(1, len(ds_full) // 20))
    train_n = len(ds_full) - val_n
    train_ds, val_ds = random_split(ds_full, [train_n, val_n],
                                    generator=torch.Generator().manual_seed(42))

    base_kw = dict(collate_fn=collate_fn, worker_init_fn=_worker_init,
                   persistent_workers=args.num_workers > 0)
    train_sampler = BucketBatchSampler(ds_full.lengths[:train_n], args.batch_size)
    train_loader  = DataLoader(train_ds, batch_sampler=train_sampler,
                               num_workers=args.num_workers, **base_kw)
    val_loader    = DataLoader(val_ds, batch_size=args.batch_size // 4,
                               shuffle=False, num_workers=0,
                               collate_fn=collate_fn, worker_init_fn=_worker_init)

    print(f"train={train_n}  val={val_n}  steps/epoch={len(train_loader)}")

    # ── Optimizer（对标 Auto-AVSR：AdamW + WarmupCosine） ─────────────────────
    # gate 保持 fp32 + lr×100（参考 v4：gate 梯度太小否则不动）
    gate_ids = set()
    for ca in model.ca_layers:
        ca.gate.data = ca.gate.data.float()
        gate_ids.add(id(ca.gate))

    pretrained_ids = set()
    for attr in ["conformer", "lm"]:
        m = getattr(model, attr, None)
        if m:
            for p in m.parameters():
                if p.requires_grad:
                    pretrained_ids.add(id(p))

    gate_params  = [p for p in model.parameters() if p.requires_grad and id(p) in gate_ids]
    pre_params   = [p for p in model.parameters() if p.requires_grad and id(p) in pretrained_ids and id(p) not in gate_ids]
    new_params   = [p for p in model.parameters() if p.requires_grad and id(p) not in pretrained_ids and id(p) not in gate_ids]

    opt = torch.optim.AdamW(
        [{"params": pre_params,  "lr": args.lr,        "weight_decay": args.weight_decay},
         {"params": new_params,  "lr": 3e-4,            "weight_decay": 0.3},
         {"params": gate_params, "lr": args.lr * 10,    "weight_decay": 0.0}],
        betas=(0.9, 0.98),   # 对标 Auto-AVSR
    )
    print(f"  optim: {len(pre_params)} pretrained + {len(new_params)} new + {len(gate_params)} gate (gate lr×10, fp32)")
    warmup_steps = int(args.warmup_epochs * len(train_loader))
    max_steps    = args.max_epochs * len(train_loader)
    print(f"warmup={warmup_steps}  max={max_steps}  lr={args.lr:.0e}")

    # ── Training loop ─────────────────────────────────────────────────────────
    step  = 0
    t_log = time.time()
    model.train()
    opt.zero_grad()

    for epoch in range(1, args.max_epochs + 1):
        for batch in train_loader:
            if batch is None:
                continue
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}

            # LR schedule（WarmupCosine）
            lr = get_lr(step, warmup_steps, max_steps, args.lr)
            scale = lr / max(args.lr, 1e-12)
            opt.param_groups[0]["lr"] = lr
            opt.param_groups[1]["lr"] = 3e-4 * scale
            opt.param_groups[2]["lr"] = args.lr * 10 * scale    # gate lr×10

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                out = model(
                    visual=batch["visual"],
                    input_ids=batch["input_ids"],
                    target_ids=batch["target_ids"],
                    video_pos=batch["video_pos"],
                    text_pos=batch["text_pos"],
                    video_mask=batch["video_mask"],
                    text_mask=batch["text_mask"],
                    last_chunk_mask=None,
                )

            loss = out["loss"] / args.grad_accum
            loss.backward()

            step += 1
            if step % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 1.0)
                opt.step()
                opt.zero_grad()

            if step % 10 == 0:
                now  = time.time()
                sps  = (now - t_log) / 10
                t_log = now
                rem  = (max_steps - step) * sps
                eta  = f"{rem/3600:.1f}h" if rem > 3600 else f"{rem/60:.0f}m"
                print(f"step {step:6d}/{max_steps} | "
                      f"ce {out['loss_ce'].item():.4f} | "
                      f"lr {lr:.2e} | {sps:.2f}s/step | eta {eta}", flush=True)
                if use_wandb:
                    wandb.log({"loss_ce": out["loss_ce"].item(), "lr": lr}, step=step)

            # Validation
            if step % args.eval_every == 0:
                metrics, examples = run_val(model, val_loader, device, tokenizer)
                if metrics:
                    print(f"\n  [val] val_ce={metrics['val_ce']:.4f}  "
                          f"val_tok_acc={metrics['val_tok_acc']:.4f}")
                    for pred, gt in examples[:3]:
                        print(f"    GT:   {gt[:80]}")
                        print(f"    PRED: {pred[:80]}")
                    if use_wandb:
                        wandb.log(metrics, step=step)

            # Save
            if step % args.save_every == 0:
                ckpt_path = out_dir / f"step_{step:06d}.pt"
                torch.save({"step": step, "model": model.state_dict(),
                            "opt": opt.state_dict()}, ckpt_path)
                print(f"  [save] {ckpt_path}")

    # Final save
    final_path = out_dir / f"step_{step:06d}_final.pt"
    torch.save({"step": step, "model": model.state_dict()}, final_path)
    print(f"Training complete. Final ckpt: {final_path}")
    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
