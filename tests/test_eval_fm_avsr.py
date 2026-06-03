import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from scripts.eval_fm_avsr import latent_metrics, shifted_condition_clip
from scripts import eval_fm_avsr
from streaminlip.fm_avsr_dataset import normalize_latent, set_norm_stats_path


class EvalFMAVSRTest(unittest.TestCase):
    def test_shifted_condition_clip_keeps_same_clip_by_default(self):
        clips = [Path("a"), Path("b"), Path("c")]

        self.assertEqual(shifted_condition_clip(clips, 1, 0), Path("b"))

    def test_shifted_condition_clip_wraps_positive_shift(self):
        clips = [Path("a"), Path("b"), Path("c")]

        self.assertEqual(shifted_condition_clip(clips, 2, 1), Path("a"))

    def test_shifted_condition_clip_rejects_empty_clips(self):
        with self.assertRaisesRegex(ValueError, "empty clips"):
            shifted_condition_clip([], 0, 1)

    def test_latent_metrics_reports_corr_and_errors(self):
        pred = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        target = pred.copy()

        metrics = latent_metrics(pred, target)

        self.assertAlmostEqual(metrics["corr"], 1.0, places=6)
        self.assertAlmostEqual(metrics["mse"], 0.0)
        self.assertAlmostEqual(metrics["mae"], 0.0)

    def test_parse_args_exposes_metrics_only_default(self):
        old_argv = sys.argv
        try:
            sys.argv = [
                "scripts/eval_fm_avsr.py",
                "--ckpt",
                "dummy.pt",
            ]
            args = eval_fm_avsr.parse_args()
        finally:
            sys.argv = old_argv

        self.assertFalse(args.metrics_only)

    def test_parse_args_loads_text_source_from_config(self):
        old_argv = sys.argv
        try:
            with tempfile.NamedTemporaryFile("w", suffix=".yaml") as f:
                f.write("text_source: text_json\nvisual_feature_name: avsr_enc_lipavsr.npy\n")
                f.flush()
                sys.argv = [
                    "scripts/eval_fm_avsr.py",
                    "--config",
                    f.name,
                    "--ckpt",
                    "dummy.pt",
                ]
                args = eval_fm_avsr.parse_args()
        finally:
            sys.argv = old_argv

        self.assertEqual(args.text_source, "text_json")
        self.assertEqual(args.visual_feature_name, "avsr_enc_lipavsr.npy")

    def test_eval_metric_target_should_use_normalized_latent(self):
        with tempfile.TemporaryDirectory() as tmp:
            stats = Path(tmp) / "stats.npz"
            np.savez(
                stats,
                mean=np.array([10.0, -10.0], dtype=np.float32),
                std=np.array([2.0, 4.0], dtype=np.float32),
            )
            set_norm_stats_path(stats)
            raw = np.array([[12.0, -6.0], [8.0, -14.0]], dtype=np.float32)
            pred_norm = normalize_latent(raw)

            metrics = latent_metrics(pred_norm, normalize_latent(raw))

        set_norm_stats_path(Path("data/processed/latent_norm_stats.npz"))
        self.assertAlmostEqual(metrics["corr"], 1.0, places=6)
        self.assertAlmostEqual(metrics["mse"], 0.0)


if __name__ == "__main__":
    unittest.main()
