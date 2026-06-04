"""
诊断实验：AV-HuBERT → Linear(768→28) → CTC loss（字符级）
无 LM，无 Conformer，验证 AV-HuBERT 特征是否可用于唇读。

收敛判据：loss < 1.0，能 greedy decode 出可辨识的字符序列。

Usage:
  python scripts/train_ctc_only.py --debug
  python scripts/train_ctc_only.py --run_name ctc_diag_lr3e-4
"""
import argparse, csv, json, os, sys, time
from pathlib import Path

_TMPDIR = Path(__file__).parent.parent / ".tmp"
_TMPDIR.mkdir(exist_ok=True)
os.environ["TMPDIR"] = str(_TMPDIR)

import torch
import torch.multiprocessing
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

torch.multiprocessing.set_sharing_strategy("file_system")
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from streaminlip.v4.visual_encoder import AVHuBERTMultiLayerExtractor, AVHUBERT_DIM
from streaminlip.v4.data.dataset import CHARS, BLANK_ID, FPS, CHUNK_SIZE

N_CHARS = BLANK_ID + 1  # 28


# ── Dataset ────────────────────────────────────────────────────────────────────

class CTCDataset(Dataset):
    def __init__(self, processed_root, split, max_frames=150, subset="train",
                 test_reserve=500, limit=None):
        self.root = Path(processed_root)
        self.max_frames = max_frames

        with open(self.root / "manifest.csv") as f:
            all_clips = [self.root / r["path"]
                         for r in csv.DictReader(f) if r["split"] == split]

        # pre_only filter
        cache = self.root / f"_pre_only_{split}.txt"
        if cache.exists():
            valid = set(cache.read_text().split())
            all_clips = [c for c in all_clips
                         if str(c.relative_to(self.root)) in valid]
        else:
            all_clips = [c for c in all_clips if (c / "avhubert_pre.npy").exists()]

        # 只保留有词时间戳的（cache 基于全量 clips，与 subset 无关）
        words_cache = self.root / f"_has_words_{split}.txt"
        if words_cache.exists():
            valid = set(words_cache.read_text().split())
        else:
            # 全量扫描一次，不受 subset 影响
            all_for_cache = all_clips if subset != "test" else (
                [self.root / r["path"]
                 for r in csv.DictReader(open(self.root / "manifest.csv"))
                 if r["split"] == split])
            valid_clips = [c for c in all_for_cache
                           if json.loads((c / "text.json").read_text()).get("words")]
            valid = set(str(c.relative_to(self.root)) for c in valid_clips)
            words_cache.write_text("\n".join(sorted(valid)))

        if subset == "test":
            all_clips = all_clips[-test_reserve:]
        else:
            all_clips = all_clips[:-test_reserve]

        self.clips = [c for c in all_clips if str(c.relative_to(self.root)) in valid]
        if limit:
            self.clips = self.clips[:limit]
        print(f"CTCDataset split={split}/{subset}  clips={len(self.clips)}")

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, idx):
        import numpy as np
        clip = self.clips[idx]
        meta  = json.loads((clip / "text.json").read_text())
        words = meta["words"]

        pre  = np.load(str(clip / "avhubert_pre.npy"), mmap_mode="r")
        T    = min(len(pre), self.max_frames)
        # snap to chunk boundary
        T    = (T // CHUNK_SIZE) * CHUNK_SIZE
        if T == 0:
            T = CHUNK_SIZE
        visual = torch.from_numpy(pre[:T].copy())   # (T, 768) float16

        # char targets: words starting before T/FPS
        t_max = T / FPS
        chars = []
        for w in words:
            if w["start"] >= t_max:
                break
            text = w["word"].lower().strip()
            if chars:
                chars.append(26)  # space
            for c in text:
                i = CHARS.find(c)
                if i >= 0:
                    chars.append(i)

        ctc_ids = torch.tensor(chars, dtype=torch.long) if chars else torch.zeros(1, dtype=torch.long)
        return {"visual": visual, "ctc_ids": ctc_ids, "T": T, "C": len(ctc_ids)}


def collate_fn(batch):
    max_T = max(b["T"] for b in batch)
    max_C = max(b["C"] for b in batch)
    B = len(batch)
    visual   = torch.zeros(B, max_T, AVHUBERT_DIM, dtype=torch.float16)
    ctc_ids  = torch.zeros(B, max_C, dtype=torch.long)
    T_lens   = torch.zeros(B, dtype=torch.long)
    C_lens   = torch.zeros(B, dtype=torch.long)
    for i, b in enumerate(batch):
        T, C = b["T"], b["C"]
        visual[i, :T]   = b["visual"]
        ctc_ids[i, :C]  = b["ctc_ids"]
        T_lens[i]       = T
        C_lens[i]       = C
    return {"visual": visual, "ctc_ids": ctc_ids, "T_lens": T_lens, "C_lens": C_lens}


# ── Model ──────────────────────────────────────────────────────────────────────

class AVHuBERTCTC(nn.Module):
    def __init__(self, avhubert_ckpt, lora_rank=16, resnet50_weights=None):
        super().__init__()
        self.encoder = AVHuBERTMultiLayerExtractor(
            avhubert_ckpt, device="cpu", lora_rank=lora_rank
        )
        self.ctc_head = nn.Linear(AVHUBERT_DIM, N_CHARS)

    def forward(self, visual, T_lens, ctc_ids, C_lens):
        last_feat, _ = self.encoder(visual)           # (B, T, 768)
        log_probs = F.log_softmax(
            self.ctc_head(last_feat.float()), dim=-1
        ).transpose(0, 1)                             # (T, B, 28)
        loss = F.ctc_loss(log_probs, ctc_ids, T_lens, C_lens,
                          blank=BLANK_ID, zero_infinity=True)
        return loss, log_probs

    def greedy_decode(self, log_probs, T_len):
        ids = log_probs[:T_len, 0].argmax(-1).tolist()
        # CTC collapse
        collapsed = [ids[0]] + [b for a, b in zip(ids, ids[1:]) if a != b]
        chars = [CHARS[i] for i in collapsed if i != BLANK_ID and i < len(CHARS)]
        return "".join(chars)


# ── main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root",        default="data/processed")
    p.add_argument("--split",            default="pretrain")
    p.add_argument("--avhubert_ckpt",    default="pretrained/self_large_vox_433h.pt")
    p.add_argument("--resnet50_weights", default="pretrained/resnet50-11ad3fa6.pth")
    p.add_argument("--lora_rank",        type=int,   default=16)
    p.add_argument("--lr",               type=float, default=3e-4)
    p.add_argument("--batch_size",       type=int,   default=512)
    p.add_argument("--max_steps",        type=int,   default=10000)
    p.add_argument("--max_frames",       type=int,   default=150)
    p.add_argument("--num_workers",      type=int,   default=12)
    p.add_argument("--log_every",        type=int,   default=50)
    p.add_argument("--run_name",         default="ctc_diag")
    p.add_argument("--output_dir",       default="runs/ctc_diag")
    p.add_argument("--debug",            action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    if args.debug:
        args.batch_size = 4
        args.max_steps  = 200
        args.num_workers = 0
        args.log_every  = 10

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path(args.output_dir) / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    ds_train = CTCDataset(args.data_root, args.split, args.max_frames,
                          subset="train", limit=None)
    ds_val   = CTCDataset(args.data_root, args.split, args.max_frames,
                          subset="test", test_reserve=500)

    train_loader = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, collate_fn=collate_fn,
                              pin_memory=True, drop_last=True)
    val_loader   = DataLoader(ds_val,   batch_size=16, shuffle=False,
                              num_workers=args.num_workers, collate_fn=collate_fn)

    model = AVHuBERTCTC(args.avhubert_ckpt, args.lora_rank, args.resnet50_weights)
    model.phase1_mode = lambda: None  # no-op
    model = model.to(device)

    trainable = [p for p in model.parameters() if p.requires_grad]
    print(f"trainable params: {sum(p.numel() for p in trainable)/1e6:.1f}M")
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=1e-2)

    step = 0
    t0 = time.time()
    for _ in range(100):
        for batch in train_loader:
            visual  = batch["visual"].to(device, dtype=torch.bfloat16)
            ctc_ids = batch["ctc_ids"].to(device)
            T_lens  = batch["T_lens"].to(device)
            C_lens  = batch["C_lens"].to(device)

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                loss, _ = model(visual, T_lens, ctc_ids, C_lens)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            optimizer.zero_grad()
            step += 1

            if step % args.log_every == 0:
                sps = (time.time() - t0) / args.log_every
                t0  = time.time()
                print(f"step {step:5d} | loss {loss.item():.4f} | {sps:.2f}s/step",
                      flush=True)

            if step % 500 == 0 or (args.debug and step % 50 == 0):
                model.eval()
                val_loss, n = 0.0, 0
                with torch.no_grad():
                    for vb in val_loader:
                        vvis = vb["visual"].to(device, dtype=torch.bfloat16)
                        vl, log_probs = model(
                            vvis, vb["T_lens"].to(device),
                            vb["ctc_ids"].to(device), vb["C_lens"].to(device)
                        )
                        val_loss += vl.item()
                        n += 1
                        if n == 1:
                            # print first sample decode
                            ref_ids = vb["ctc_ids"][0, :vb["C_lens"][0]].tolist()
                            ref = "".join(CHARS[i] for i in ref_ids if i < len(CHARS))
                            hyp = model.greedy_decode(log_probs, vb["T_lens"][0].item())
                            print(f"  REF: {ref[:80]}")
                            print(f"  HYP: {hyp[:80]}")
                if n > 0:
                    print(f"  [val] loss={val_loss/n:.4f}", flush=True)
                torch.save({"step": step, "encoder": model.encoder.state_dict(),
                            "ctc_head": model.ctc_head.state_dict()},
                           str(out_dir / f"step_{step:06d}.pt"))
                model.train()

            if step >= args.max_steps:
                print("Done.")
                return


if __name__ == "__main__":
    main()
