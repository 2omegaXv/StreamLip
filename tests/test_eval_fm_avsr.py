import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import torch

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

    def test_latent_metrics_can_skip_prompt_frames(self):
        pred = np.array([[100.0], [2.0], [3.0]], dtype=np.float32)
        target = np.array([[-100.0], [2.0], [3.0]], dtype=np.float32)

        metrics = latent_metrics(pred, target, start_frame=1)

        self.assertAlmostEqual(metrics["mse"], 0.0)
        self.assertAlmostEqual(metrics["corr"], 1.0)

    def test_slice_latent_for_wav_output_skips_prompt_frames(self):
        lat = np.arange(5 * 2, dtype=np.float32).reshape(5, 2)

        out = eval_fm_avsr.slice_latent_for_wav_output(lat, start_frame=3)

        np.testing.assert_array_equal(out, lat[3:])

    def test_slice_latent_for_wav_output_keeps_at_least_one_frame(self):
        lat = np.arange(3 * 2, dtype=np.float32).reshape(3, 2)

        out = eval_fm_avsr.slice_latent_for_wav_output(lat, start_frame=10)

        np.testing.assert_array_equal(out, lat[-1:])

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

    def test_parse_args_loads_residual_base_ckpt_from_config(self):
        old_argv = sys.argv
        try:
            with tempfile.NamedTemporaryFile("w", suffix=".yaml") as f:
                f.write("residual_base_ckpt: runs/fm_avsr/base.pt\n")
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

        self.assertEqual(args.residual_base_ckpt, "runs/fm_avsr/base.pt")

    def test_parse_args_loads_allow_partial_resume_from_config(self):
        old_argv = sys.argv
        try:
            with tempfile.NamedTemporaryFile("w", suffix=".yaml") as f:
                f.write("allow_partial_resume: true\n")
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

        self.assertTrue(args.allow_partial_resume)

    def test_parse_args_loads_audio_prompt_learned_pool_from_config(self):
        old_argv = sys.argv
        try:
            with tempfile.NamedTemporaryFile("w", suffix=".yaml") as f:
                f.write("audio_prompt_learned_pool_cond: true\n")
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

        self.assertTrue(args.audio_prompt_learned_pool_cond)

    def test_parse_args_prefers_val_clip_list_from_train_config(self):
        old_argv = sys.argv
        try:
            with tempfile.NamedTemporaryFile("w", suffix=".yaml") as f:
                f.write("clip_list: train.txt\nval_clip_list: val.txt\n")
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

        self.assertEqual(args.clip_list, "val.txt")

    def test_parse_args_cli_clip_list_overrides_config_val_clip_list(self):
        old_argv = sys.argv
        try:
            with tempfile.NamedTemporaryFile("w", suffix=".yaml") as f:
                f.write("clip_list: train.txt\nval_clip_list: val.txt\n")
                f.flush()
                sys.argv = [
                    "scripts/eval_fm_avsr.py",
                    "--config",
                    f.name,
                    "--ckpt",
                    "dummy.pt",
                    "--clip_list",
                    "manual.txt",
                ]
                args = eval_fm_avsr.parse_args()
        finally:
            sys.argv = old_argv

        self.assertEqual(args.clip_list, "manual.txt")

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

    def test_residual_base_latent_is_computed_for_sample_eval_path(self):
        class FakeResidualBase:
            def __init__(self):
                self.calls = 0

            def reconstruct_from_cond(self, v_down, h_down, spk_t, **kwargs):
                self.calls += 1
                return torch.ones(v_down.shape[0], v_down.shape[1], 512)

        residual_base = FakeResidualBase()
        v_down = torch.zeros(1, 3, 768)
        h_down = torch.zeros(1, 3, 960)
        spk_t = torch.zeros(1, 256)

        base = eval_fm_avsr.ensure_residual_base_latent(
            None,
            residual_base,
            v_down,
            h_down,
            spk_t,
        )

        self.assertEqual(residual_base.calls, 1)
        torch.testing.assert_close(base, torch.ones(1, 3, 512))


if __name__ == "__main__":
    unittest.main()
