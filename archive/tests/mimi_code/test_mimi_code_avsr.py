import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.build_mimi_code_cache import cache_path_for_clip, read_clip_list
from scripts.train_mimi_code_avsr import (
    ARMimiCodeHead,
    MimiCodeAVSRDataset,
    assert_disjoint_clip_lists,
    build_optimizer,
    load_training_checkpoint,
    collate_code_batch,
    condition_mode_needs_text_hidden,
    masked_code_loss_and_acc,
    train_loader_drop_last,
)


class MimiCodeAVSRTest(unittest.TestCase):
    def test_read_clip_list_resolves_relative_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_root = root / "processed"
            clip = data_root / "pretrain" / "video" / "00001"
            clip.mkdir(parents=True)
            clip_list = root / "clips.txt"
            clip_list.write_text("pretrain/video/00001\n")

            clips = read_clip_list(clip_list, data_root=data_root)

            self.assertEqual(clips, [clip])

    def test_cache_path_for_clip_preserves_last_three_parts(self):
        clip = Path("/data/processed/pretrain/video_a/00007")
        out = cache_path_for_clip(Path("/cache"), clip)

        self.assertEqual(out, Path("/cache/pretrain/video_a/00007/mimi_codes.npz"))

    def test_collate_code_batch_pads_and_masks_code_tokens(self):
        batch = [
            {
                "enc": np.ones((4, 768), dtype=np.float32),
                "speaker": np.ones((256,), dtype=np.float32),
                "codes": np.array([1, 2], dtype=np.int64),
                "h_lm": np.ones((3, 960), dtype=np.float32),
            },
            {
                "enc": np.full((6, 768), 2, dtype=np.float32),
                "speaker": np.full((256,), 2, dtype=np.float32),
                "codes": np.array([3, 4, 5], dtype=np.int64),
                "h_lm": np.full((5, 960), 2, dtype=np.float32),
            },
        ]

        out = collate_code_batch(batch)

        self.assertEqual(out["enc"].shape, (2, 6, 768))
        self.assertEqual(out["codes"].shape, (2, 3))
        self.assertEqual(out["h_lm"].shape, (2, 5, 960))
        np.testing.assert_array_equal(out["lens_L"], np.array([3, 5]))
        np.testing.assert_array_equal(out["code_lens"], np.array([2, 3]))
        np.testing.assert_array_equal(out["codes"][0], np.array([1, 2, 0]))

    def test_dataset_loads_text_hidden_from_selected_text_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_root = root / "processed"
            clip = data_root / "pretrain" / "video" / "00001"
            cache_clip = root / "cache" / "pretrain" / "video" / "00001"
            clip.mkdir(parents=True)
            cache_clip.mkdir(parents=True)
            np.save(clip / "avsr_enc.npy", np.ones((4, 768), dtype=np.float32))
            np.save(clip / "speaker_emb.npy", np.ones((256,), dtype=np.float32))
            np.save(clip / "smollm2_h.npy", np.full((2, 960), 1, dtype=np.float32))
            np.save(clip / "smollm2_h_text_json.npy", np.full((3, 960), 2, dtype=np.float32))
            np.savez(cache_clip / "mimi_codes.npz", codes=np.array([[[7, 8, 9]]], dtype=np.int64))
            clip_list = root / "clips.txt"
            clip_list.write_text("pretrain/video/00001\n")

            ds = MimiCodeAVSRDataset(
                clip_list=clip_list,
                data_root=data_root,
                code_cache_root=root / "cache",
                load_text_hidden=True,
                text_source="text_json",
            )

            item = ds[0]
            self.assertEqual(item["h_lm"].shape, (3, 960))
            self.assertEqual(float(item["h_lm"][0, 0]), 2.0)

    def test_masked_code_loss_accepts_label_smoothing(self):
        import torch

        logits = torch.zeros(1, 2, 4)
        codes = torch.tensor([[1, 2]])
        lengths = torch.tensor([1])

        loss, acc = masked_code_loss_and_acc(logits, codes, lengths, label_smoothing=0.1)

        self.assertGreater(float(loss), 0.0)
        self.assertEqual(float(acc), 0.0)

    def test_assert_disjoint_clip_lists_rejects_overlap(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            train = root / "train.txt"
            val = root / "val.txt"
            train.write_text("pretrain/a/00001\npretrain/b/00002\n")
            val.write_text("pretrain/b/00002\n")

            with self.assertRaisesRegex(ValueError, "overlap"):
                assert_disjoint_clip_lists(train, val, data_root=root / "processed")

    def test_ar_mimi_code_head_uses_previous_code_history(self):
        import torch

        torch.manual_seed(0)
        model = ARMimiCodeHead(dim=32, n_layers=1, n_heads=4, condition_mode="video_spk").eval()
        enc = torch.randn(1, 6, 768)
        spk = torch.randn(1, 256)
        codes_a = torch.tensor([[1, 2, 3]])
        codes_b = torch.tensor([[4, 5, 6]])

        with torch.no_grad():
            logits_a = model(enc, spk, codes_a)
            logits_b = model(enc, spk, codes_b)

        self.assertEqual(logits_a.shape, (1, 3, 2048))
        self.assertFalse(torch.allclose(logits_a, logits_b))

    def test_ar_mimi_code_head_uses_text_hidden_condition(self):
        import torch

        torch.manual_seed(0)
        model = ARMimiCodeHead(
            dim=32,
            n_layers=1,
            n_heads=4,
            condition_mode="video_spk_text",
        ).eval()
        enc = torch.randn(1, 6, 768)
        spk = torch.randn(1, 256)
        codes = torch.tensor([[1, 2, 3]])
        h_a = torch.zeros(1, 4, 960)
        h_b = torch.ones(1, 4, 960)
        lens = torch.tensor([4])

        with torch.no_grad():
            logits_a = model(enc, spk, codes, h_lm=h_a, lens_L=lens)
            logits_b = model(enc, spk, codes, h_lm=h_b, lens_L=lens)

        self.assertEqual(logits_a.shape, (1, 3, 2048))
        self.assertFalse(torch.allclose(logits_a, logits_b))

    def test_ar_mimi_code_head_cross_attends_video_frames(self):
        import torch

        torch.manual_seed(0)
        model = ARMimiCodeHead(
            dim=32,
            n_layers=1,
            n_heads=4,
            condition_mode="video_spk_crossattn",
        ).eval()
        enc_a = torch.randn(1, 8, 768)
        enc_b = enc_a.clone()
        enc_b[:, 1::2] = enc_b[:, 1::2] + 2.0
        spk = torch.randn(1, 256)
        codes = torch.tensor([[1, 2, 3, 4]])

        with torch.no_grad():
            logits_a = model(enc_a, spk, codes)
            logits_b = model(enc_b, spk, codes)

        self.assertEqual(logits_a.shape, (1, 4, 2048))
        self.assertFalse(torch.allclose(logits_a, logits_b))

    def test_ar_mimi_code_head_cross_attends_text_and_masks_padding(self):
        import torch

        torch.manual_seed(0)
        model = ARMimiCodeHead(
            dim=32,
            n_layers=1,
            n_heads=4,
            condition_mode="video_spk_text_crossattn",
        ).eval()
        enc = torch.randn(1, 6, 768)
        spk = torch.randn(1, 256)
        codes = torch.tensor([[1, 2, 3]])
        lens = torch.tensor([2])
        h_a = torch.zeros(1, 4, 960)
        h_b = h_a.clone()
        h_b[:, 1] = 1.0
        h_pad_changed = h_a.clone()
        h_pad_changed[:, 2:] = 10.0

        with torch.no_grad():
            logits_a = model(enc, spk, codes, h_lm=h_a, lens_L=lens)
            logits_b = model(enc, spk, codes, h_lm=h_b, lens_L=lens)
            logits_pad_changed = model(enc, spk, codes, h_lm=h_pad_changed, lens_L=lens)

        self.assertEqual(logits_a.shape, (1, 3, 2048))
        self.assertFalse(torch.allclose(logits_a, logits_b))
        self.assertTrue(torch.allclose(logits_a, logits_pad_changed))

    def test_ar_mimi_code_head_cross_attends_video_and_text_contexts(self):
        import torch

        torch.manual_seed(0)
        model = ARMimiCodeHead(
            dim=32,
            n_layers=1,
            n_heads=4,
            condition_mode="video_spk_video_text_crossattn",
        ).eval()
        enc_a = torch.randn(1, 8, 768)
        enc_b = enc_a.clone()
        enc_b[:, 1::2] = enc_b[:, 1::2] + 2.0
        spk = torch.randn(1, 256)
        codes = torch.tensor([[1, 2, 3, 4]])
        lens = torch.tensor([2])
        h_a = torch.zeros(1, 4, 960)
        h_b = h_a.clone()
        h_b[:, 1] = 1.0
        h_pad_changed = h_a.clone()
        h_pad_changed[:, 2:] = 10.0

        with torch.no_grad():
            logits_a = model(enc_a, spk, codes, h_lm=h_a, lens_L=lens)
            logits_video_changed = model(enc_b, spk, codes, h_lm=h_a, lens_L=lens)
            logits_text_changed = model(enc_a, spk, codes, h_lm=h_b, lens_L=lens)
            logits_pad_changed = model(enc_a, spk, codes, h_lm=h_pad_changed, lens_L=lens)

        self.assertEqual(logits_a.shape, (1, 4, 2048))
        self.assertFalse(torch.allclose(logits_a, logits_video_changed))
        self.assertFalse(torch.allclose(logits_a, logits_text_changed))
        self.assertTrue(torch.allclose(logits_a, logits_pad_changed))

    def test_small_debug_dataset_keeps_partial_training_batch(self):
        self.assertFalse(train_loader_drop_last(dataset_len=32, batch_size=64))
        self.assertTrue(train_loader_drop_last(dataset_len=128, batch_size=64))

    def test_text_condition_modes_load_text_hidden(self):
        self.assertTrue(condition_mode_needs_text_hidden("video_spk_text"))
        self.assertTrue(condition_mode_needs_text_hidden("video_spk_text_crossattn"))
        self.assertTrue(condition_mode_needs_text_hidden("video_spk_video_text_crossattn"))
        self.assertFalse(condition_mode_needs_text_hidden("video_spk"))

    def test_build_optimizer_uses_configured_weight_decay(self):
        import torch

        model = torch.nn.Linear(2, 3)
        opt = build_optimizer(model, lr=1e-3, weight_decay=0.05)

        self.assertEqual(opt.param_groups[0]["weight_decay"], 0.05)

    def test_load_training_checkpoint_restores_model_optimizer_and_step(self):
        import torch

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ckpt.pt"
            src = torch.nn.Linear(2, 3)
            src_opt = build_optimizer(src, lr=1e-3, weight_decay=0.05)
            with torch.no_grad():
                src.weight.fill_(2.0)
            src_opt.zero_grad()
            src(torch.ones(1, 2)).sum().backward()
            src_opt.step()
            torch.save({"model": src.state_dict(), "optimizer": src_opt.state_dict(), "step": 7}, path)
            dst = torch.nn.Linear(2, 3)
            dst_opt = build_optimizer(dst, lr=1e-3, weight_decay=0.05)

            step = load_training_checkpoint(path, dst, dst_opt, device="cpu")

            self.assertEqual(step, 7)
            self.assertTrue(torch.allclose(src.weight, dst.weight))
            self.assertEqual(dst_opt.state_dict()["state"].keys(), src_opt.state_dict()["state"].keys())

    def test_load_training_checkpoint_can_skip_optimizer_restore(self):
        import torch

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ckpt.pt"
            src = torch.nn.Linear(2, 3)
            src_opt = build_optimizer(src, lr=1e-3, weight_decay=0.05)
            with torch.no_grad():
                src.weight.fill_(2.0)
            src_opt.zero_grad()
            src(torch.ones(1, 2)).sum().backward()
            src_opt.step()
            torch.save({"model": src.state_dict(), "optimizer": src_opt.state_dict(), "step": 7}, path)
            dst = torch.nn.Linear(2, 3)
            dst_opt = build_optimizer(dst, lr=1e-4, weight_decay=0.05)

            step = load_training_checkpoint(
                path,
                dst,
                dst_opt,
                device="cpu",
                restore_optimizer=False,
            )

            self.assertEqual(step, 7)
            self.assertTrue(torch.allclose(src.weight, dst.weight))
            self.assertEqual(dst_opt.param_groups[0]["lr"], 1e-4)
            self.assertEqual(dst_opt.state_dict()["state"], {})


if __name__ == "__main__":
    unittest.main()
