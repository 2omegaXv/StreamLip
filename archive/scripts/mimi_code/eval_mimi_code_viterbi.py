import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent))
import train_mimi_code_avsr as train
from build_mimi_code_cache import cache_path_for_clip, read_clip_list


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base_run", required=True)
    p.add_argument("--base_ckpt", required=True)
    p.add_argument("--output_json", default=None)
    p.add_argument("--topk", type=int, default=5)
    p.add_argument("--alphas", default="0,0.2,0.4,0.6,0.8,1.0")
    p.add_argument("--bigram_smoothing", type=float, default=0.05)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--max_val_clips", type=int, default=0)
    p.add_argument("--progress_every", type=int, default=100)
    return p.parse_args()


def load_base_args(base_run):
    return type("Args", (), json.loads((Path(base_run) / "config.json").read_text()))()


def build_bigram_logp(clip_list, data_root, code_cache_root, codebook, smoothing=0.05):
    vocab = train.VOCAB_SIZE
    counts = np.ones((vocab + 1, vocab), dtype=np.float64) * float(smoothing)
    clips = read_clip_list(clip_list, data_root=data_root)
    for clip in clips:
        codes = np.load(cache_path_for_clip(code_cache_root, clip))["codes"][0, codebook].astype(np.int64)
        prev = vocab
        for code in codes:
            counts[prev, int(code)] += 1.0
            prev = int(code)
    return torch.from_numpy(np.log(counts / counts.sum(axis=1, keepdims=True)).astype(np.float32))


def viterbi_topk_decode(top_ids, top_logits, bigram_logp, bos_id, alpha):
    code_len, topk = top_ids.shape
    if code_len == 0:
        return top_ids.new_empty((0,))
    if alpha == 0:
        return top_ids[:, 0]
    dp = top_logits[0] + alpha * bigram_logp[bos_id, top_ids[0]]
    back = []
    for t in range(1, code_len):
        trans = bigram_logp[top_ids[t - 1].view(topk, 1), top_ids[t].view(1, topk)]
        score = dp.view(topk, 1) + alpha * trans + top_logits[t].view(1, topk)
        best, arg = score.max(dim=0)
        dp = best
        back.append(arg)
    cur = int(dp.argmax().item())
    path = [cur]
    for arg in reversed(back):
        cur = int(arg[cur].item())
        path.append(cur)
    path.reverse()
    return top_ids[torch.arange(code_len, device=top_ids.device), torch.tensor(path, device=top_ids.device)]


def build_base_model(base_args, ckpt_path, device):
    if base_args.architecture != "ar":
        raise ValueError("Viterbi eval expects an AR base model")
    model = train.ARMimiCodeHead(
        base_args.dim,
        base_args.n_layers,
        base_args.n_heads,
        base_args.condition_mode,
        base_args.dropout,
    ).to(device)
    step = train.load_training_checkpoint(ckpt_path, model, optimizer=None, device=device, restore_optimizer=False)
    model.eval()
    return model, step


def main():
    args = parse_args()
    alphas = [float(x) for x in args.alphas.split(",") if x.strip()]
    base_args = load_base_args(args.base_run)
    train.assert_disjoint_clip_lists(base_args.clip_list, base_args.val_clip_list, data_root=base_args.data_root)
    start = time.time()
    print("Building bigram prior...", flush=True)
    bigram_logp = build_bigram_logp(
        base_args.clip_list,
        Path(base_args.data_root),
        Path(base_args.code_cache_root),
        int(base_args.codebook),
        smoothing=args.bigram_smoothing,
    )
    print(f"Bigram prior ready in {time.time() - start:.1f}s", flush=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    bigram_logp = bigram_logp.to(device)
    model, step = build_base_model(base_args, args.base_ckpt, device)
    load_text_hidden = train.condition_mode_needs_text_hidden(base_args.condition_mode)
    val_ds = train.MimiCodeAVSRDataset(
        base_args.val_clip_list,
        base_args.data_root,
        base_args.code_cache_root,
        base_args.codebook,
        limit=args.max_val_clips,
        load_text_hidden=load_text_hidden,
        text_source=base_args.text_source,
    )
    val_loader = DataLoader(
        val_ds,
        shuffle=False,
        batch_size=args.batch_size,
        collate_fn=train.collate_code_batch,
        num_workers=0,
    )
    correct = {alpha: 0 for alpha in alphas}
    total = 0
    base_correct = 0
    candidate_correct = 0
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            enc = torch.from_numpy(batch["enc"]).to(device)
            spk = torch.from_numpy(batch["speaker"]).to(device)
            codes = torch.from_numpy(batch["codes"]).to(device)
            code_lens = torch.from_numpy(batch["code_lens"]).to(device)
            h_lm = torch.from_numpy(batch["h_lm"]).to(device) if batch["h_lm"] is not None else None
            lens_L = torch.from_numpy(batch["lens_L"]).to(device) if batch["lens_L"] is not None else None
            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
                logits = model(enc, spk, codes, h_lm=h_lm, lens_L=lens_L).float()
            for b in range(logits.shape[0]):
                code_len = int(code_lens[b].item())
                target = codes[b, :code_len]
                top_logits, top_ids = logits[b, :code_len].topk(args.topk, dim=-1)
                total += int(code_len)
                base_correct += int((top_ids[:, 0] == target).sum().item())
                candidate_correct += int((top_ids == target.unsqueeze(-1)).any(dim=-1).sum().item())
                for alpha in alphas:
                    pred = viterbi_topk_decode(
                        top_ids,
                        top_logits,
                        bigram_logp,
                        bos_id=train.VOCAB_SIZE,
                        alpha=alpha,
                    )
                    correct[alpha] += int((pred == target).sum().item())
            if args.progress_every and (i + 1) % args.progress_every == 0:
                print(f"eval batches {i + 1} | tokens {total} | base {base_correct / max(total, 1):.4f}", flush=True)
    result = {
        "base_run": args.base_run,
        "base_ckpt": args.base_ckpt,
        "base_step": step,
        "topk": args.topk,
        "bigram_smoothing": args.bigram_smoothing,
        "val_clips": len(val_ds),
        "tokens": total,
        "base_acc": base_correct / max(total, 1),
        "candidate_acc": candidate_correct / max(total, 1),
        "alpha_acc": {str(alpha): correct[alpha] / max(total, 1) for alpha in alphas},
        "best_alpha": max(alphas, key=lambda a: correct[a]),
        "best_acc": max(correct.values()) / max(total, 1),
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    if args.output_json:
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output_json).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
