"""
SmolLM2 LRS3 域内适配（Causal LM fine-tune）。

在 LRS3 转写文本上做 next-token prediction，让模型适应 TED 演讲语言分布。
保留通用能力：低 LR + 少 epoch + 验证集 perplexity 监控。

Usage:
  python scripts/finetune_lm.py
  python scripts/finetune_lm.py --epochs 2 --lr 3e-5 --batch 32
"""
import argparse, math, time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm


# ── Dataset ──────────────────────────────────────────────────────────────────

class TextDataset(Dataset):
    def __init__(self, path: str, tokenizer, max_len: int = 256):
        lines = Path(path).read_text().splitlines()
        lines = [l.strip() for l in lines if l.strip()]
        self.tok     = tokenizer
        self.max_len = max_len
        # 把多行拼成 chunk，用 EOS 分隔，避免短句梯度被 padding 稀释
        all_ids = []
        for line in lines:
            ids = tokenizer.encode(line, add_special_tokens=False)
            ids.append(tokenizer.eos_token_id)
            all_ids.extend(ids)
        # 切成固定长度 chunk
        self.chunks = [
            all_ids[i:i + max_len]
            for i in range(0, len(all_ids) - max_len, max_len)
        ]
        print(f"TextDataset: {len(lines)} lines → {len(all_ids):,} tokens → "
              f"{len(self.chunks):,} chunks (len={max_len})")

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        ids = torch.tensor(self.chunks[idx], dtype=torch.long)
        return ids[:-1], ids[1:]   # input, target


# ── Training ──────────────────────────────────────────────────────────────────

def run_epoch(model, loader, optimizer, scheduler, device, train: bool):
    model.train(train)
    total_loss = total_tok = 0
    ctx = torch.no_grad() if not train else torch.enable_grad()
    with ctx:
        for x, y in tqdm(loader, desc="train" if train else "val ", leave=False):
            x, y = x.to(device), y.to(device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(x).logits                          # (B, L, V)
                loss   = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    y.reshape(-1),
                )
            if train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 1.0)
                optimizer.step()
                if scheduler: scheduler.step()
            total_loss += loss.item() * y.numel()
            total_tok  += y.numel()
    return total_loss / total_tok   # per-token CE = log perplexity


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", default="pretrained/smollm2-360m")
    p.add_argument("--data",       default="data/processed/lrs3_text.txt")
    p.add_argument("--output",     default="pretrained/smollm2-360m-lrs3-lower")
    p.add_argument("--max_len",    type=int,   default=256)
    p.add_argument("--batch",      type=int,   default=128)
    p.add_argument("--epochs",     type=int,   default=3)
    p.add_argument("--lr",         type=float, default=1.2e-4)
    p.add_argument("--val_frac",   type=float, default=0.02)
    p.add_argument("--gpu",        type=int,   default=0)
    p.add_argument("--workers",    type=int,   default=8)
    p.add_argument("--save_each_epoch", action="store_true",
                   help="每 epoch 额外保存一份 output-ep{N}/（sweep 时不加，单次跑时加）")
    args = p.parse_args()

    device = f"cuda:{args.gpu}"

    print("Loading tokenizer & model …")
    tok   = AutoTokenizer.from_pretrained(args.model_path)
    tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16
    ).to(device)
    n = sum(p.numel() for p in model.parameters())
    print(f"Model: {n/1e6:.0f}M params, device={device}")

    ds = TextDataset(args.data, tok, max_len=args.max_len)
    n_val   = max(1, int(len(ds) * args.val_frac))
    n_train = len(ds) - n_val
    train_ds, val_ds = random_split(ds, [n_train, n_val],
                                    generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                              num_workers=args.workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch, shuffle=False,
                              num_workers=args.workers, pin_memory=True)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=0.01, betas=(0.9, 0.95)
    )
    total_steps = len(train_loader) * args.epochs
    warmup      = total_steps // 10
    scheduler   = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr, total_steps=total_steps,
        pct_start=warmup / total_steps, anneal_strategy="cos",
    )

    best_val = float("inf")
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_ce = run_epoch(model, train_loader, optimizer, scheduler, device, train=True)
        val_ce   = run_epoch(model, val_loader,   optimizer, None,      device, train=False)
        ppl      = math.exp(val_ce)
        print(f"Epoch {epoch}/{args.epochs} | "
              f"train_ce={train_ce:.4f}  val_ce={val_ce:.4f}  val_ppl={ppl:.1f} | "
              f"{time.time()-t0:.0f}s")
        if val_ce < best_val:
            best_val = val_ce
            model.save_pretrained(args.output)
            tok.save_pretrained(args.output)
            print(f"  → best updated → {args.output}")
        if args.save_each_epoch:
            ep_output = f"{args.output}-ep{epoch}"
            model.save_pretrained(ep_output)
            tok.save_pretrained(ep_output)
            print(f"  → saved to {ep_output}")

    print(f"\nDone. Best val_ppl={math.exp(best_val):.1f}  best → {args.output}")


if __name__ == "__main__":
    main()
