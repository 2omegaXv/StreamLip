import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))
from build_mimi_code_cache import cache_path_for_clip, read_clip_list


VOCAB_SIZE = 2048


def assert_disjoint_clip_lists(train_clip_list, val_clip_list, data_root=None):
    train = {str(p) for p in read_clip_list(train_clip_list, data_root=data_root)}
    val = {str(p) for p in read_clip_list(val_clip_list, data_root=data_root)}
    overlap = train & val
    if overlap:
        examples = ", ".join(sorted(overlap)[:5])
        raise ValueError(
            f"train/val clip lists overlap: {len(overlap)} clips; examples: {examples}"
        )


def explicit_cli_keys(argv=None):
    argv = list(sys.argv if argv is None else argv)
    keys = set()
    for item in argv[1:]:
        if item.startswith("--"):
            keys.add(item[2:].split("=", 1)[0].replace("-", "_"))
    return keys


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=None)
    p.add_argument("--data_root", default="data/processed")
    p.add_argument("--code_cache_root", required=False, default="data/mimi_code_cache")
    p.add_argument("--clip_list", required=False, default=None)
    p.add_argument("--val_clip_list", required=False, default=None)
    p.add_argument("--output_dir", default="runs/mimi_code_avsr")
    p.add_argument("--run_name", default="mimi_code_avsr_v1")
    p.add_argument("--codebook", type=int, default=0)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--max_steps", type=int, default=5000)
    p.add_argument("--eval_every", type=int, default=500)
    p.add_argument("--save_every", type=int, default=1000)
    p.add_argument("--n_layers", type=int, default=4)
    p.add_argument("--dim", type=int, default=512)
    p.add_argument("--n_heads", type=int, default=8)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--label_smoothing", type=float, default=0.0)
    p.add_argument("--condition_mode", choices=["video_spk", "video_only"], default="video_spk")
    p.add_argument("--debug", action="store_true")
    cli_keys = explicit_cli_keys()
    args = p.parse_args()
    if args.config:
        import yaml
        cfg = yaml.safe_load(Path(args.config).read_text()) or {}
        valid = set(vars(args))
        unknown = sorted(set(cfg) - valid)
        if unknown:
            raise ValueError(f"Unknown config keys in {args.config}: {unknown}")
        for k, v in cfg.items():
            if k in cli_keys and k != "config":
                continue
            setattr(args, k, v)
    return args


class MimiCodeAVSRDataset(Dataset):
    def __init__(self, clip_list, data_root, code_cache_root, codebook=0, limit=0):
        self.clips = read_clip_list(clip_list, data_root=data_root, n=limit)
        self.code_cache_root = Path(code_cache_root)
        self.codebook = int(codebook)

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, idx):
        clip = self.clips[idx]
        enc = np.load(clip / "avsr_enc.npy").astype("float32")
        spk = np.load(clip / "speaker_emb.npy").astype("float32")
        codes_npz = cache_path_for_clip(self.code_cache_root, clip)
        codes = np.load(codes_npz)["codes"][0, self.codebook].astype("int64")
        return {"enc": enc, "speaker": spk, "codes": codes}


def collate_code_batch(batch):
    max_t = max(item["enc"].shape[0] for item in batch)
    max_c = max(item["codes"].shape[0] for item in batch)
    bsz = len(batch)
    enc = np.zeros((bsz, max_t, 768), dtype=np.float32)
    spk = np.stack([item["speaker"] for item in batch]).astype(np.float32)
    codes = np.zeros((bsz, max_c), dtype=np.int64)
    enc_lens = np.zeros((bsz,), dtype=np.int64)
    code_lens = np.zeros((bsz,), dtype=np.int64)
    for i, item in enumerate(batch):
        t = item["enc"].shape[0]
        c = item["codes"].shape[0]
        enc[i, :t] = item["enc"]
        codes[i, :c] = item["codes"]
        enc_lens[i] = t
        code_lens[i] = c
    return {"enc": enc, "speaker": spk, "codes": codes, "enc_lens": enc_lens, "code_lens": code_lens}


class TransformerBlock(nn.Module):
    def __init__(self, dim, n_heads, mlp_ratio=4, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, n_heads, batch_first=True, dropout=dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mlp_ratio, dim),
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x, key_padding_mask=None):
        y, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x), key_padding_mask=key_padding_mask)
        x = x + self.drop(y)
        x = x + self.drop(self.ffn(self.norm2(x)))
        return x


class MimiCodeHead(nn.Module):
    def __init__(self, dim=512, n_layers=4, n_heads=8, condition_mode="video_spk", dropout=0.0):
        super().__init__()
        self.condition_mode = condition_mode
        self.enc_proj = nn.Linear(768, dim)
        self.spk_proj = nn.Linear(256, dim)
        self.blocks = nn.ModuleList([TransformerBlock(dim, n_heads, dropout=dropout) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(dim)
        self.out = nn.Linear(dim, VOCAB_SIZE)

    def forward(self, enc, spk, code_len):
        bsz, _, _ = enc.shape
        h = self.enc_proj(enc[:, ::2, :])
        if h.shape[1] > code_len:
            h = h[:, :code_len]
        elif h.shape[1] < code_len:
            pad = h.new_zeros(bsz, code_len - h.shape[1], h.shape[2])
            h = torch.cat([h, pad], dim=1)
        if self.condition_mode == "video_spk":
            h = h + self.spk_proj(spk).unsqueeze(1)
        pos = sinusoidal_positions(code_len, h.shape[-1], h.device, h.dtype)
        h = h + pos
        for block in self.blocks:
            h = block(h)
        return self.out(self.norm(h))


def sinusoidal_positions(length, dim, device, dtype):
    half = dim // 2
    pos = torch.arange(length, device=device, dtype=torch.float32).unsqueeze(1)
    freqs = torch.exp(
        -math.log(10000) * torch.arange(half, device=device, dtype=torch.float32) / half
    ).unsqueeze(0)
    emb = torch.cat([(pos * freqs).sin(), (pos * freqs).cos()], dim=-1)
    if emb.shape[-1] < dim:
        emb = F.pad(emb, (0, dim - emb.shape[-1]))
    return emb.to(dtype).unsqueeze(0)


def masked_code_loss_and_acc(logits, codes, code_lens, label_smoothing=0.0):
    bsz, t, vocab = logits.shape
    pos = torch.arange(t, device=logits.device).unsqueeze(0)
    mask = pos < code_lens.unsqueeze(1)
    loss = F.cross_entropy(
        logits.reshape(-1, vocab),
        codes[:, :t].reshape(-1),
        reduction="none",
        label_smoothing=label_smoothing,
    )
    loss = (loss.reshape(bsz, t) * mask).sum() / mask.sum().clamp_min(1)
    pred = logits.argmax(dim=-1)
    acc = ((pred == codes[:, :t]) & mask).sum().float() / mask.sum().clamp_min(1)
    return loss, acc


@torch.no_grad()
def evaluate(model, loader, device, max_batches=0, label_smoothing=0.0):
    model.eval()
    loss_sum = 0.0
    acc_sum = 0.0
    n = 0
    for i, batch in enumerate(loader):
        enc = torch.from_numpy(batch["enc"]).to(device)
        spk = torch.from_numpy(batch["speaker"]).to(device)
        codes = torch.from_numpy(batch["codes"]).to(device)
        code_lens = torch.from_numpy(batch["code_lens"]).to(device)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
            logits = model(enc, spk, codes.shape[1])
            loss, acc = masked_code_loss_and_acc(logits.float(), codes, code_lens, label_smoothing)
        loss_sum += float(loss.cpu())
        acc_sum += float(acc.cpu())
        n += 1
        if max_batches and i + 1 >= max_batches:
            break
    model.train()
    return {"loss": loss_sum / max(n, 1), "acc": acc_sum / max(n, 1), "n_batches": n}


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path(args.output_dir) / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(vars(args), indent=2, sort_keys=True) + "\n")
    if args.val_clip_list:
        assert_disjoint_clip_lists(args.clip_list, args.val_clip_list, data_root=args.data_root)

    train_ds = MimiCodeAVSRDataset(
        args.clip_list, args.data_root, args.code_cache_root, args.codebook, limit=32 if args.debug else 0
    )
    val_ds = MimiCodeAVSRDataset(
        args.val_clip_list, args.data_root, args.code_cache_root, args.codebook, limit=32 if args.debug else 0
    )
    loader_kw = dict(batch_size=args.batch_size, collate_fn=collate_code_batch, num_workers=args.num_workers, drop_last=True)
    if args.num_workers > 0:
        loader_kw.update(dict(persistent_workers=True, prefetch_factor=2))
    train_loader = DataLoader(train_ds, shuffle=True, **loader_kw)
    val_loader = DataLoader(val_ds, shuffle=False, batch_size=args.batch_size, collate_fn=collate_code_batch, num_workers=0)

    model = MimiCodeHead(args.dim, args.n_layers, args.n_heads, args.condition_mode, args.dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.1)
    metrics_path = out_dir / "metrics.csv"
    val_path = out_dir / "val_metrics.csv"
    with metrics_path.open("w", newline="") as mf, val_path.open("w", newline="") as vf:
        metrics = csv.DictWriter(mf, fieldnames=["step", "loss", "acc", "lr", "elapsed_seconds"])
        vals = csv.DictWriter(vf, fieldnames=["step", "val_loss", "val_acc", "train_loss", "train_acc", "n_batches", "elapsed_seconds"])
        metrics.writeheader()
        vals.writeheader()
        step = 0
        start = time.time()
        while step < args.max_steps:
            for batch in train_loader:
                enc = torch.from_numpy(batch["enc"]).to(device)
                spk = torch.from_numpy(batch["speaker"]).to(device)
                codes = torch.from_numpy(batch["codes"]).to(device)
                code_lens = torch.from_numpy(batch["code_lens"]).to(device)
                with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
                    logits = model(enc, spk, codes.shape[1])
                    loss, acc = masked_code_loss_and_acc(
                        logits.float(), codes, code_lens, args.label_smoothing
                    )
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                step += 1
                elapsed = time.time() - start
                metrics.writerow({
                    "step": step,
                    "loss": f"{float(loss.detach().cpu()):.8f}",
                    "acc": f"{float(acc.detach().cpu()):.8f}",
                    "lr": f"{args.lr:.10e}",
                    "elapsed_seconds": f"{elapsed:.4f}",
                })
                mf.flush()
                if step % 10 == 0:
                    print(f"step {step}/{args.max_steps} | ce {float(loss.detach().cpu()):.4f} | acc {float(acc.detach().cpu()):.4f}", flush=True)
                if step % args.eval_every == 0:
                    val_m = evaluate(model, val_loader, device, max_batches=10, label_smoothing=args.label_smoothing)
                    train_m = evaluate(model, train_loader, device, max_batches=10, label_smoothing=args.label_smoothing)
                    vals.writerow({
                        "step": step,
                        "val_loss": f"{val_m['loss']:.8f}",
                        "val_acc": f"{val_m['acc']:.8f}",
                        "train_loss": f"{train_m['loss']:.8f}",
                        "train_acc": f"{train_m['acc']:.8f}",
                        "n_batches": val_m["n_batches"],
                        "elapsed_seconds": f"{elapsed:.4f}",
                    })
                    vf.flush()
                    print(f"eval step {step} | val_ce {val_m['loss']:.4f} | val_acc {val_m['acc']:.4f} | train_acc {train_m['acc']:.4f}", flush=True)
                if step % args.save_every == 0:
                    torch.save({"model": model.state_dict(), "step": step, "args": vars(args)}, out_dir / f"step_{step:06d}.pt")
                if step >= args.max_steps:
                    break
    torch.save({"model": model.state_dict(), "step": step, "args": vars(args)}, out_dir / f"step_{step:06d}.pt")


if __name__ == "__main__":
    main()
