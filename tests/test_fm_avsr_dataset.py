import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from streaminlip.fm_avsr_dataset import (
    FMAVSRDataset,
    build_word_timestamp_lm_indices,
    read_clip_text,
    validate_latent_frame_rate,
)
from scripts.train_fm_avsr import (
    aggregate_sample_metrics,
    combine_training_losses,
    crop_batch_to_latent_window,
    explicit_cli_keys,
    masked_mse_loss,
    parse_args,
    prepare_conditions,
)


class FMAVSRDatasetTest(unittest.TestCase):
    def test_read_clip_text_can_use_text_json_words(self):
        with tempfile.TemporaryDirectory() as tmp:
            clip = Path(tmp)
            (clip / "avsr_text.txt").write_text("BAD ASR\n")
            (clip / "text.json").write_text(
                '{"words":[{"word":"GOOD"},{"word":"TEXT"}]}'
            )

            self.assertEqual(read_clip_text(clip, "text_json"), "GOOD TEXT")
            self.assertEqual(read_clip_text(clip, "avsr"), "BAD ASR")

    def test_dataset_text_json_source_uses_matching_hidden_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            clip = root / "pretrain" / "spk" / "00001"
            clip.mkdir(parents=True)
            (clip / "avsr_text.txt").write_text("BAD ASR\n")
            (clip / "text.json").write_text(
                '{"words":[{"word":"GOOD"},{"word":"TEXT"}]}'
            )
            np.save(clip / "avsr_enc.npy", np.ones((4, 768), dtype=np.float32))
            np.savez(clip / "latent.npz", latent=np.ones((2, 512), dtype=np.float32))
            np.save(clip / "speaker_emb.npy", np.ones((256,), dtype=np.float32))
            np.save(clip / "smollm2_h.npy", np.ones((1, 960), dtype=np.float16))
            np.save(clip / "smollm2_h_text_json.npy", np.full((3, 960), 2, dtype=np.float16))
            clip_list = root / "clips.txt"
            clip_list.write_text("pretrain/spk/00001\n")

            ds = FMAVSRDataset(
                str(root),
                clip_list=str(clip_list),
                text_source="text_json",
            )
            item = ds[0]

            self.assertEqual(item["text"], "GOOD TEXT")
            self.assertEqual(item["h_lm"].shape[0], 3)
            self.assertAlmostEqual(float(item["h_lm"][0, 0]), 2.0)

    def test_reject_25hz_latent(self):
        lat = np.arange(10 * 2, dtype=np.float32).reshape(10, 2)
        with self.assertRaisesRegex(ValueError, "25Hz"):
            validate_latent_frame_rate(lat, enc_len=10, clip="dummy")

    def test_keep_existing_12p5hz_latent(self):
        lat = np.arange(5 * 2, dtype=np.float32).reshape(5, 2)
        validated = validate_latent_frame_rate(lat, enc_len=10, clip="dummy")
        np.testing.assert_array_equal(validated, lat)

    def test_crop_batch_to_latent_window_keeps_enc_and_latent_aligned(self):
        rng = np.random.default_rng(0)
        enc = np.arange(2 * 20 * 1, dtype=np.float32).reshape(2, 20, 1)
        latent = np.arange(2 * 10 * 1, dtype=np.float32).reshape(2, 10, 1)
        lengths = np.array([10, 3], dtype=np.int64)
        enc_crop, latent_crop, crop_lengths = crop_batch_to_latent_window(
            enc, latent, lengths, crop_ta=4, rng=rng
        )

        self.assertEqual(latent_crop.shape, (2, 4, 1))
        self.assertEqual(enc_crop.shape, (2, 8, 1))
        np.testing.assert_array_equal(crop_lengths, np.array([4, 3]))
        for b in range(2):
            if crop_lengths[b] < 4:
                np.testing.assert_array_equal(latent_crop[b, crop_lengths[b] :, 0], 0)
                continue
            start = int(latent_crop[b, 0, 0] - latent[b, 0, 0])
            self.assertEqual(start % 10, start)
            np.testing.assert_array_equal(
                enc_crop[b, :, 0],
                enc[b, start * 2 : start * 2 + 8, 0],
            )

    def test_masked_mse_loss_ignores_padded_latent_frames(self):
        import torch

        pred = torch.tensor([[[1.0], [10.0], [10.0]]])
        target = torch.tensor([[[0.0], [0.0], [0.0]]])
        lengths = torch.tensor([1])

        loss = masked_mse_loss(pred, target, lengths)

        self.assertAlmostEqual(loss.item(), 1.0)

    def test_aggregate_sample_metrics_ignores_padding(self):
        import torch

        pred = torch.tensor([[[1.0], [2.0], [100.0]], [[3.0], [4.0], [100.0]]])
        target = torch.tensor([[[1.0], [2.0], [-100.0]], [[3.0], [4.0], [-100.0]]])
        lengths = torch.tensor([2, 2])

        metrics = aggregate_sample_metrics(pred, target, lengths)

        self.assertAlmostEqual(metrics["mse"], 0.0)
        self.assertAlmostEqual(metrics["corr"], 1.0)

    def test_combine_training_losses_can_disable_fm_loss(self):
        import torch

        loss = combine_training_losses(
            loss_fm=torch.tensor(100.0),
            loss_recon=torch.tensor(2.0),
            loss_sample_recon=torch.tensor(3.0),
            loss_denoise=torch.tensor(5.0),
            loss_fm_weight=0.0,
            lambda_recon=1.0,
            lambda_sample_recon=0.5,
            lambda_denoise=0.25,
        )

        self.assertAlmostEqual(loss.item(), 4.75)

    def test_explicit_cli_keys_maps_flags_to_arg_names(self):
        keys = explicit_cli_keys([
            "scripts/train_fm_avsr.py",
            "--config",
            "cfg.yaml",
            "--max_steps",
            "1",
            "--run_name=debug",
            "--no_wandb",
        ])

        self.assertIn("config", keys)
        self.assertIn("max_steps", keys)
        self.assertIn("run_name", keys)
        self.assertIn("no_wandb", keys)

    def test_condition_mode_cli_overrides_deprecated_no_text_config(self):
        old_argv = sys.argv
        try:
            with tempfile.NamedTemporaryFile("w", suffix=".yaml") as f:
                f.write("no_text_cond: true\n")
                f.flush()
                sys.argv = [
                    "scripts/train_fm_avsr.py",
                    "--config",
                    f.name,
                    "--condition_mode",
                    "both",
                ]

                args = parse_args()

            self.assertEqual(args.condition_mode, "both")
        finally:
            sys.argv = old_argv

    def test_prepare_conditions_can_ablate_video_or_text(self):
        import torch

        batch = {
            "enc": np.ones((1, 6, 768), dtype=np.float32),
            "latent": np.zeros((1, 3, 512), dtype=np.float32),
            "latent_lens": np.array([3], dtype=np.int64),
            "speaker": np.ones((1, 256), dtype=np.float32),
            "h_lm": np.ones((1, 2, 960), dtype=np.float32),
            "lens_L": np.array([2], dtype=np.int64),
        }

        v_text, h_text, *_ = prepare_conditions(
            batch, "cpu", condition_mode="text_only"
        )
        v_video, h_video, *_ = prepare_conditions(
            batch, "cpu", condition_mode="video_only"
        )

        self.assertTrue(torch.all(v_text == 0))
        self.assertGreater(h_text.abs().sum().item(), 0)
        self.assertGreater(v_video.abs().sum().item(), 0)
        self.assertTrue(torch.all(h_video == 0))

    def test_prepare_conditions_can_shuffle_text_with_fixed_permutation(self):
        batch = {
            "enc": np.ones((2, 4, 768), dtype=np.float32),
            "latent": np.zeros((2, 2, 512), dtype=np.float32),
            "latent_lens": np.array([2, 2], dtype=np.int64),
            "speaker": np.ones((2, 256), dtype=np.float32),
            "h_lm": np.stack([
                np.full((1, 960), 1, dtype=np.float32),
                np.full((1, 960), 2, dtype=np.float32),
            ]),
            "lens_L": np.array([1, 1], dtype=np.int64),
        }

        _, h_down, *_ = prepare_conditions(
            batch,
            "cpu",
            condition_mode="shuffle_text",
            text_perm=np.array([1, 0], dtype=np.int64),
        )

        self.assertAlmostEqual(float(h_down[0, 0, 0]), 2.0)
        self.assertAlmostEqual(float(h_down[1, 0, 0]), 1.0)

    def test_build_word_timestamp_lm_indices_uses_word_timing(self):
        class Tok:
            def encode(self, text, add_special_tokens=False):
                return [101] if "one" in text else [102]
            def decode(self, tokens):
                return "tok"

        words = [
            {"word": "ONE", "start": 0.0, "end": 0.4},
            {"word": "TWO", "start": 1.2, "end": 1.6},
        ]

        idx = build_word_timestamp_lm_indices(
            "ONE TWO", words, Tok(), T_a=20, latent_hz=10.0
        )

        self.assertEqual(idx[1], 1)
        self.assertEqual(idx[7], 1)
        self.assertEqual(idx[13], 2)

    def test_build_word_timestamp_lm_indices_interpolates_extra_avsr_words(self):
        class Tok:
            def encode(self, text, add_special_tokens=False):
                return [1]
            def decode(self, tokens):
                return "tok"

        words = [
            {"word": "A", "start": 0.0, "end": 0.2},
            {"word": "B", "start": 0.6, "end": 0.8},
        ]

        idx = build_word_timestamp_lm_indices(
            "A EXTRA B", words, Tok(), T_a=10, latent_hz=10.0
        )

        self.assertEqual(idx[1], 1)
        self.assertEqual(idx[4], 2)
        self.assertEqual(idx[7], 3)

    def test_prepare_conditions_can_use_word_timestamp_lm_indices(self):
        batch = {
            "enc": np.ones((1, 6, 768), dtype=np.float32),
            "latent": np.zeros((1, 3, 512), dtype=np.float32),
            "latent_lens": np.array([3], dtype=np.int64),
            "speaker": np.ones((1, 256), dtype=np.float32),
            "h_lm": np.stack([
                np.stack([
                    np.full((960,), 0, dtype=np.float32),
                    np.full((960,), 10, dtype=np.float32),
                    np.full((960,), 20, dtype=np.float32),
                ])
            ]),
            "lens_L": np.array([3], dtype=np.int64),
            "lm_idx": np.array([[1, 1, 2]], dtype=np.int64),
        }

        _, h_down, *_ = prepare_conditions(
            batch,
            "cpu",
            text_alignment_mode="word_timestamps",
        )

        self.assertAlmostEqual(float(h_down[0, 0, 0]), 10.0)
        self.assertAlmostEqual(float(h_down[0, 1, 0]), 10.0)
        self.assertAlmostEqual(float(h_down[0, 2, 0]), 20.0)

    def test_prepare_conditions_can_add_ctc_logprob_extra_condition(self):
        import torch

        batch = {
            "enc": np.ones((1, 6, 768), dtype=np.float32),
            "latent": np.zeros((1, 3, 512), dtype=np.float32),
            "latent_lens": np.array([3], dtype=np.int64),
            "speaker": np.ones((1, 256), dtype=np.float32),
            "h_lm": None,
            "lens_L": None,
            "ctc_cond": np.arange(1 * 6 * 4, dtype=np.float32).reshape(1, 6, 4),
        }

        *_, extra, ctc_ids, ctc_probs = prepare_conditions(
            batch,
            "cpu",
            condition_mode="video_only",
            ctc_condition_mode="logprob",
        )

        self.assertEqual(tuple(extra.shape), (1, 3, 4))
        self.assertIsNone(ctc_ids)
        self.assertIsNone(ctc_probs)
        self.assertTrue(torch.allclose(extra[0, 0].float(), torch.tensor([0., 1., 2., 3.])))
        self.assertTrue(torch.allclose(extra[0, 1].float(), torch.tensor([8., 9., 10., 11.])))

    def test_prepare_conditions_can_add_ctc_topk_token_condition(self):
        import torch

        batch = {
            "enc": np.ones((1, 6, 768), dtype=np.float32),
            "latent": np.zeros((1, 3, 512), dtype=np.float32),
            "latent_lens": np.array([3], dtype=np.int64),
            "speaker": np.ones((1, 256), dtype=np.float32),
            "h_lm": None,
            "lens_L": None,
            "ctc_topk_ids": np.arange(1 * 6 * 2, dtype=np.int64).reshape(1, 6, 2),
            "ctc_topk_probs": np.ones((1, 6, 2), dtype=np.float32),
        }

        *_, extra, ctc_ids, ctc_probs = prepare_conditions(
            batch,
            "cpu",
            condition_mode="video_only",
            ctc_condition_mode="topk",
        )

        self.assertIsNone(extra)
        self.assertEqual(tuple(ctc_ids.shape), (1, 3, 2))
        self.assertEqual(tuple(ctc_probs.shape), (1, 3, 2))
        self.assertTrue(torch.equal(ctc_ids[0, 0].long(), torch.tensor([0, 1])))
        self.assertTrue(torch.equal(ctc_ids[0, 1].long(), torch.tensor([4, 5])))


if __name__ == "__main__":
    unittest.main()
