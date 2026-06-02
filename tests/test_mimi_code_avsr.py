import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.build_mimi_code_cache import cache_path_for_clip, read_clip_list
from scripts.train_mimi_code_avsr import (
    assert_disjoint_clip_lists,
    collate_code_batch,
    masked_code_loss_and_acc,
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
            },
            {
                "enc": np.full((6, 768), 2, dtype=np.float32),
                "speaker": np.full((256,), 2, dtype=np.float32),
                "codes": np.array([3, 4, 5], dtype=np.int64),
            },
        ]

        out = collate_code_batch(batch)

        self.assertEqual(out["enc"].shape, (2, 6, 768))
        self.assertEqual(out["codes"].shape, (2, 3))
        np.testing.assert_array_equal(out["code_lens"], np.array([2, 3]))
        np.testing.assert_array_equal(out["codes"][0], np.array([1, 2, 0]))

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


if __name__ == "__main__":
    unittest.main()
