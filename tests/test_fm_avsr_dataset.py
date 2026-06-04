import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from streaminlip.fm_avsr_dataset import (
    FMAVSRDataset,
    build_word_timestamp_lm_indices,
    collate_fn,
    compute_log_rms_energy,
    read_clip_text,
    smollm2_hidden_path,
    validate_latent_frame_rate,
)
from scripts.train_fm_avsr import (
    aggregate_sample_metrics,
    combine_training_losses,
    crop_batch_to_latent_window,
    energy_extra_dim,
    explicit_cli_keys,
    load_fm_head_state,
    residual_base_extra_dim,
    compose_residual_prediction,
    compose_endpoint_prediction,
    masked_corr_loss,
    masked_mse_loss,
    masked_prompt_timbre_stats_loss,
    masked_sample_corr_loss,
    masked_timbre_stats_loss,
    parse_args,
    project_latent_to_pca_target,
    prepare_conditions,
    predict_energy_condition,
)


class FMAVSRDatasetTest(unittest.TestCase):
    def test_dataset_loads_custom_timbre_condition_and_collates_it(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            clip = root / "pretrain" / "spk" / "00001"
            clip.mkdir(parents=True)
            (clip / "avsr_text.txt").write_text("TIMBRE\n")
            np.save(clip / "avsr_enc.npy", np.ones((4, 768), dtype=np.float32))
            np.savez(clip / "latent.npz", latent=np.ones((2, 512), dtype=np.float32))
            np.save(clip / "speaker_emb.npy", np.ones((256,), dtype=np.float32))
            np.save(clip / "smollm2_h.npy", np.ones((1, 960), dtype=np.float16))
            np.save(clip / "timbre_cond.npy", np.arange(1024, dtype=np.float32))
            clip_list = root / "clips.txt"
            clip_list.write_text("pretrain/spk/00001\n")

            ds = FMAVSRDataset(
                str(root),
                clip_list=str(clip_list),
                timbre_condition_name="timbre_cond.npy",
            )
            item = ds[0]
            batch = collate_fn([item])

            self.assertEqual(item["timbre_cond"].shape, (1024,))
            self.assertEqual(batch["timbre_cond"].shape, (1, 1024))
            self.assertAlmostEqual(float(batch["timbre_cond"][0, 17]), 17.0)

    def test_dataset_builds_audio_prompt_from_normalized_prefix_and_collates_it(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            clip = root / "pretrain" / "spk" / "00001"
            clip.mkdir(parents=True)
            (clip / "avsr_text.txt").write_text("PROMPT\n")
            np.save(clip / "avsr_enc.npy", np.ones((8, 768), dtype=np.float32))
            latent = np.arange(4 * 512, dtype=np.float32).reshape(4, 512)
            np.savez(clip / "latent.npz", latent=latent)
            np.save(clip / "speaker_emb.npy", np.ones((256,), dtype=np.float32))
            np.save(clip / "smollm2_h.npy", np.ones((1, 960), dtype=np.float16))
            np.savez(
                root / "latent_norm_stats.npz",
                mean=np.ones((512,), dtype=np.float32),
                std=np.full((512,), 2.0, dtype=np.float32),
            )
            clip_list = root / "clips.txt"
            clip_list.write_text("pretrain/spk/00001\n")

            ds = FMAVSRDataset(
                str(root),
                clip_list=str(clip_list),
                audio_prompt_frames=3,
            )
            item = ds[0]
            batch = collate_fn([item])

            expected = (latent[:3] - 1.0) / 2.0
            self.assertEqual(item["audio_prompt"].shape, (3, 512))
            self.assertEqual(batch["audio_prompt"].shape, (1, 3, 512))
            np.testing.assert_allclose(batch["audio_prompt"][0], expected)

    def test_dataset_can_build_audio_prompt_from_next_same_parent_clip(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            np.savez(
                root / "latent_norm_stats.npz",
                mean=np.zeros((512,), dtype=np.float32),
                std=np.ones((512,), dtype=np.float32),
            )
            clip_a = root / "pretrain" / "spk" / "00001"
            clip_b = root / "pretrain" / "spk" / "00002"
            for clip, value in [(clip_a, 1.0), (clip_b, 7.0)]:
                clip.mkdir(parents=True)
                (clip / "avsr_text.txt").write_text("PROMPT\n")
                np.save(clip / "avsr_enc.npy", np.ones((8, 768), dtype=np.float32))
                np.savez(
                    clip / "latent.npz",
                    latent=np.full((4, 512), value, dtype=np.float32),
                )
                np.save(clip / "speaker_emb.npy", np.ones((256,), dtype=np.float32))
                np.save(clip / "smollm2_h.npy", np.ones((1, 960), dtype=np.float16))
            clip_list = root / "clips.txt"
            clip_list.write_text("pretrain/spk/00001\npretrain/spk/00002\n")

            ds = FMAVSRDataset(
                str(root),
                clip_list=str(clip_list),
                audio_prompt_frames=3,
                audio_prompt_ref_mode="same_parent_next",
            )
            item = ds[0]

            self.assertEqual(item["audio_prompt"].shape, (3, 512))
            np.testing.assert_allclose(item["audio_prompt"], np.full((3, 512), 7.0))

    def test_read_clip_text_can_use_text_json_words(self):
        with tempfile.TemporaryDirectory() as tmp:
            clip = Path(tmp)
            (clip / "avsr_text.txt").write_text("BAD ASR\n")
            (clip / "text.json").write_text(
                '{"words":[{"word":"GOOD"},{"word":"TEXT"}]}'
            )

            self.assertEqual(read_clip_text(clip, "text_json"), "GOOD TEXT")
            self.assertEqual(read_clip_text(clip, "avsr"), "BAD ASR")

    def test_lipavsr_text_source_uses_lipavsr_text_and_hidden_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            clip = root / "pretrain" / "spk" / "00001"
            clip.mkdir(parents=True)
            (clip / "avsr_text.txt").write_text("OLD ASR\n")
            (clip / "avsr_text_lipavsr.txt").write_text("NEW LIP AVSR\n")
            np.save(clip / "avsr_enc.npy", np.ones((4, 768), dtype=np.float32))
            np.savez(clip / "latent.npz", latent=np.ones((2, 512), dtype=np.float32))
            np.save(clip / "speaker_emb.npy", np.ones((256,), dtype=np.float32))
            np.save(clip / "smollm2_h.npy", np.ones((1, 960), dtype=np.float16))
            np.save(clip / "smollm2_h_lipavsr.npy", np.full((2, 960), 3, dtype=np.float16))
            clip_list = root / "clips.txt"
            clip_list.write_text("pretrain/spk/00001\n")

            ds = FMAVSRDataset(
                str(root),
                clip_list=str(clip_list),
                text_source="lipavsr",
            )
            item = ds[0]

            self.assertEqual(read_clip_text(clip, "lipavsr"), "NEW LIP AVSR")
            self.assertEqual(smollm2_hidden_path(clip, "lipavsr").name, "smollm2_h_lipavsr.npy")
            self.assertEqual(item["text"], "NEW LIP AVSR")
            self.assertEqual(item["h_lm"].shape[0], 2)
            self.assertAlmostEqual(float(item["h_lm"][0, 0]), 3.0)

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
            sf.write(clip / "audio.wav", np.ones(2400, dtype=np.float32), 24000)
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

    def test_dataset_can_load_custom_visual_feature_name(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            clip = root / "pretrain" / "spk" / "00001"
            clip.mkdir(parents=True)
            (clip / "avsr_text.txt").write_text("CUSTOM VISUAL\n")
            np.save(clip / "avsr_enc_lipavsr.npy", np.full((4, 768), 3, dtype=np.float32))
            np.savez(clip / "latent.npz", latent=np.ones((2, 512), dtype=np.float32))
            np.save(clip / "speaker_emb.npy", np.ones((256,), dtype=np.float32))
            np.save(clip / "smollm2_h.npy", np.ones((1, 960), dtype=np.float16))
            clip_list = root / "clips.txt"
            clip_list.write_text("pretrain/spk/00001\n")

            ds = FMAVSRDataset(
                str(root),
                clip_list=str(clip_list),
                visual_feature_name="avsr_enc_lipavsr.npy",
            )
            item = ds[0]

            self.assertEqual(item["enc"].shape, (4, 768))
            self.assertAlmostEqual(float(item["enc"][0, 0]), 3.0)

    def test_dataset_loads_log_rms_energy_aligned_to_latent(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            clip = root / "pretrain" / "spk" / "00001"
            clip.mkdir(parents=True)
            (clip / "avsr_text.txt").write_text("ENERGY\n")
            np.save(clip / "avsr_enc.npy", np.ones((4, 768), dtype=np.float32))
            np.savez(clip / "latent.npz", latent=np.ones((2, 512), dtype=np.float32))
            np.save(clip / "speaker_emb.npy", np.ones((256,), dtype=np.float32))
            np.save(clip / "smollm2_h.npy", np.ones((1, 960), dtype=np.float16))
            sf.write(clip / "audio.wav", np.ones(2400, dtype=np.float32), 24000)
            clip_list = root / "clips.txt"
            clip_list.write_text("pretrain/spk/00001\n")

            ds = FMAVSRDataset(str(root), clip_list=str(clip_list), load_energy=True)
            item = ds[0]

            self.assertEqual(item["energy"].shape, (2, 1))
            np.testing.assert_allclose(
                item["energy"][:, 0],
                compute_log_rms_energy(np.ones(2400, dtype=np.float32), 2),
                atol=1e-6,
            )

    def test_dataset_default_does_not_require_audio_wav(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            clip = root / "pretrain" / "spk" / "00001"
            clip.mkdir(parents=True)
            (clip / "avsr_text.txt").write_text("NO ENERGY\n")
            np.save(clip / "avsr_enc.npy", np.ones((4, 768), dtype=np.float32))
            np.savez(clip / "latent.npz", latent=np.ones((2, 512), dtype=np.float32))
            np.save(clip / "speaker_emb.npy", np.ones((256,), dtype=np.float32))
            np.save(clip / "smollm2_h.npy", np.ones((1, 960), dtype=np.float16))
            clip_list = root / "clips.txt"
            clip_list.write_text("pretrain/spk/00001\n")

            ds = FMAVSRDataset(str(root), clip_list=str(clip_list))
            item = ds[0]

            self.assertNotIn("energy", item)

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

    def test_crop_batch_to_latent_window_keeps_energy_aligned(self):
        rng = np.random.default_rng(0)
        enc = np.arange(1 * 20 * 1, dtype=np.float32).reshape(1, 20, 1)
        latent = np.arange(1 * 10 * 1, dtype=np.float32).reshape(1, 10, 1)
        energy = np.arange(1 * 10 * 1, dtype=np.float32).reshape(1, 10, 1) + 100
        lengths = np.array([10], dtype=np.int64)

        _, latent_crop, energy_crop, crop_lengths = crop_batch_to_latent_window(
            enc, latent, lengths, crop_ta=4, rng=rng, energy_np=energy
        )

        start = int(latent_crop[0, 0, 0])
        np.testing.assert_array_equal(energy_crop[0, :, 0], energy[0, start:start + 4, 0])
        np.testing.assert_array_equal(crop_lengths, np.array([4]))

    def test_masked_mse_loss_ignores_padded_latent_frames(self):
        import torch

        pred = torch.tensor([[[1.0], [10.0], [10.0]]])
        target = torch.tensor([[[0.0], [0.0], [0.0]]])
        lengths = torch.tensor([1])

        loss = masked_mse_loss(pred, target, lengths)

        self.assertAlmostEqual(loss.item(), 1.0)

    def test_masked_mse_loss_can_skip_prompt_frames(self):
        import torch

        pred = torch.tensor([[[10.0], [1.0], [2.0]]])
        target = torch.zeros_like(pred)
        lengths = torch.tensor([3])

        loss = masked_mse_loss(pred, target, lengths, start_frame=1)

        self.assertAlmostEqual(loss.item(), 2.5)

    def test_masked_corr_loss_is_zero_for_perfect_valid_frames(self):
        import torch

        pred = torch.tensor([[[1.0], [2.0], [100.0]]])
        target = torch.tensor([[[1.0], [2.0], [-100.0]]])
        lengths = torch.tensor([2])

        loss = masked_corr_loss(pred, target, lengths)

        self.assertAlmostEqual(loss.item(), 0.0, places=5)

    def test_masked_corr_loss_penalizes_anti_correlation(self):
        import torch

        pred = torch.tensor([[[1.0], [2.0], [3.0]]])
        target = torch.tensor([[[3.0], [2.0], [1.0]]])
        lengths = torch.tensor([3])

        loss = masked_corr_loss(pred, target, lengths)

        self.assertAlmostEqual(loss.item(), 2.0, places=5)

    def test_masked_corr_loss_can_skip_prompt_frames(self):
        import torch

        pred = torch.tensor([[[100.0], [1.0], [2.0], [3.0]]])
        target = torch.tensor([[[-100.0], [1.0], [2.0], [3.0]]])
        lengths = torch.tensor([4])

        loss = masked_corr_loss(pred, target, lengths, start_frame=1)

        self.assertAlmostEqual(loss.item(), 0.0, places=5)

    def test_masked_corr_loss_has_finite_grad_for_constant_prediction(self):
        import torch

        pred = torch.zeros(1, 3, 1, requires_grad=True)
        target = torch.tensor([[[1.0], [2.0], [3.0]]])
        lengths = torch.tensor([3])

        loss = masked_corr_loss(pred, target, lengths)
        loss.backward()

        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(pred.grad).all())

    def test_masked_sample_corr_loss_averages_per_clip_correlation(self):
        import torch

        pred = torch.tensor([
            [[0.0], [1.0], [2.0]],
            [[0.0], [1.0], [2.0]],
        ])
        target = torch.tensor([
            [[0.0], [1.0], [2.0]],
            [[0.0], [-1.0], [-2.0]],
        ])
        lengths = torch.tensor([3, 3])

        loss = masked_sample_corr_loss(pred, target, lengths, start_frame=1)

        # Per-clip correlations are +1 and -1, so mean corr is 0 and loss is 1.
        self.assertAlmostEqual(loss.item(), 1.0, places=5)

    def test_masked_timbre_stats_loss_matches_valid_post_prompt_mean_and_std(self):
        import torch

        pred = torch.tensor([[[100.0], [1.0], [3.0], [1000.0]]])
        target = torch.tensor([[[-100.0], [2.0], [4.0], [-1000.0]]])
        lengths = torch.tensor([3])

        loss = masked_timbre_stats_loss(pred, target, lengths, start_frame=1)

        # Valid post-prompt frames are [1, 3] and [2, 4]. Means differ by 1,
        # unbiased=False stds match, and padded frame 3 must be ignored.
        self.assertAlmostEqual(loss.item(), 1.0)

    def test_masked_prompt_timbre_stats_loss_matches_prompt_mean_and_std(self):
        import torch

        pred = torch.tensor([[[100.0], [1.0], [3.0], [1000.0]]])
        prompt = torch.tensor([[[2.0], [4.0]]])
        lengths = torch.tensor([3])

        loss = masked_prompt_timbre_stats_loss(pred, prompt, lengths, start_frame=1)

        # Valid predicted post-prompt frames are [1, 3], while prompt frames are
        # [2, 4]. Means differ by 1, stds match, and padded frame 3 is ignored.
        self.assertAlmostEqual(loss.item(), 1.0)

    def test_project_latent_to_pca_target_can_disable_projection(self):
        import torch

        lat = torch.randn(2, 3, 4)

        out = project_latent_to_pca_target(lat, None, None, 0)

        self.assertIs(out, lat)

    def test_project_latent_to_pca_target_reconstructs_low_rank_latent(self):
        import torch

        lat = torch.tensor([[[3.0, 4.0]]])
        mean = torch.tensor([[1.0, 1.0]])
        components = torch.tensor([[1.0, 0.0], [0.0, 1.0]])

        out = project_latent_to_pca_target(lat, mean, components, 1)

        self.assertTrue(torch.equal(out, torch.tensor([[[3.0, 1.0]]])))

    def test_aggregate_sample_metrics_ignores_padding(self):
        import torch

        pred = torch.tensor([[[1.0], [2.0], [100.0]], [[3.0], [4.0], [100.0]]])
        target = torch.tensor([[[1.0], [2.0], [-100.0]], [[3.0], [4.0], [-100.0]]])
        lengths = torch.tensor([2, 2])

        metrics = aggregate_sample_metrics(pred, target, lengths)

        self.assertAlmostEqual(metrics["mse"], 0.0)
        self.assertAlmostEqual(metrics["corr"], 1.0)

    def test_aggregate_sample_metrics_can_skip_prompt_frames(self):
        import torch

        pred = torch.tensor([[[100.0], [2.0], [3.0]]])
        target = torch.tensor([[[-100.0], [2.0], [3.0]]])
        lengths = torch.tensor([3])

        metrics = aggregate_sample_metrics(pred, target, lengths, start_frame=1)

        self.assertAlmostEqual(metrics["mse"], 0.0)
        self.assertAlmostEqual(metrics["corr"], 1.0)

    def test_combine_training_losses_can_disable_fm_loss(self):
        import torch

        loss = combine_training_losses(
            loss_fm=torch.tensor(100.0),
            loss_recon=torch.tensor(2.0),
            loss_sample_recon=torch.tensor(3.0),
            loss_denoise=torch.tensor(5.0),
            loss_energy=torch.tensor(7.0),
            loss_recon_corr=torch.tensor(11.0),
            loss_sample_corr=torch.tensor(19.0),
            loss_recon_pca=torch.tensor(13.0),
            loss_timbre_stats=torch.tensor(17.0),
            loss_prompt_timbre_stats=torch.tensor(23.0),
            loss_fm_weight=0.0,
            lambda_recon=1.0,
            lambda_sample_recon=0.5,
            lambda_denoise=0.25,
            lambda_energy=0.0,
            lambda_recon_corr=0.0,
            lambda_sample_corr=0.0,
            lambda_recon_pca=0.0,
            lambda_timbre_stats=0.0,
            lambda_prompt_timbre_stats=0.0,
        )

        self.assertAlmostEqual(loss.item(), 4.75)

    def test_combine_training_losses_can_include_recon_corr_loss(self):
        import torch

        loss = combine_training_losses(
            loss_fm=torch.tensor(0.0),
            loss_recon=torch.tensor(2.0),
            loss_sample_recon=torch.tensor(0.0),
            loss_denoise=torch.tensor(0.0),
            loss_energy=torch.tensor(0.0),
            loss_recon_corr=torch.tensor(3.0),
            loss_sample_corr=torch.tensor(0.0),
            loss_recon_pca=torch.tensor(7.0),
            loss_timbre_stats=torch.tensor(0.0),
            loss_prompt_timbre_stats=torch.tensor(0.0),
            loss_fm_weight=0.0,
            lambda_recon=1.0,
            lambda_sample_recon=0.0,
            lambda_denoise=0.0,
            lambda_energy=0.0,
            lambda_recon_corr=0.5,
            lambda_sample_corr=0.0,
            lambda_recon_pca=0.0,
            lambda_timbre_stats=0.0,
            lambda_prompt_timbre_stats=0.0,
        )

        self.assertAlmostEqual(loss.item(), 3.5)

    def test_combine_training_losses_can_include_sample_corr_loss(self):
        import torch

        loss = combine_training_losses(
            loss_fm=torch.tensor(0.0),
            loss_recon=torch.tensor(2.0),
            loss_sample_recon=torch.tensor(0.0),
            loss_denoise=torch.tensor(0.0),
            loss_energy=torch.tensor(0.0),
            loss_recon_corr=torch.tensor(0.0),
            loss_sample_corr=torch.tensor(3.0),
            loss_recon_pca=torch.tensor(0.0),
            loss_timbre_stats=torch.tensor(0.0),
            loss_prompt_timbre_stats=torch.tensor(0.0),
            loss_fm_weight=0.0,
            lambda_recon=1.0,
            lambda_sample_recon=0.0,
            lambda_denoise=0.0,
            lambda_energy=0.0,
            lambda_recon_corr=0.0,
            lambda_sample_corr=0.5,
            lambda_recon_pca=0.0,
            lambda_timbre_stats=0.0,
            lambda_prompt_timbre_stats=0.0,
        )

        self.assertAlmostEqual(loss.item(), 3.5)

    def test_combine_training_losses_can_include_pca_recon_loss(self):
        import torch

        loss = combine_training_losses(
            loss_fm=torch.tensor(0.0),
            loss_recon=torch.tensor(2.0),
            loss_sample_recon=torch.tensor(0.0),
            loss_denoise=torch.tensor(0.0),
            loss_energy=torch.tensor(0.0),
            loss_recon_corr=torch.tensor(0.0),
            loss_sample_corr=torch.tensor(0.0),
            loss_recon_pca=torch.tensor(3.0),
            loss_timbre_stats=torch.tensor(0.0),
            loss_prompt_timbre_stats=torch.tensor(0.0),
            loss_fm_weight=0.0,
            lambda_recon=1.0,
            lambda_sample_recon=0.0,
            lambda_denoise=0.0,
            lambda_energy=0.0,
            lambda_recon_corr=0.0,
            lambda_sample_corr=0.0,
            lambda_recon_pca=0.25,
            lambda_timbre_stats=0.0,
            lambda_prompt_timbre_stats=0.0,
        )

        self.assertAlmostEqual(loss.item(), 2.75)

    def test_combine_training_losses_can_include_energy_loss(self):
        import torch

        loss = combine_training_losses(
            loss_fm=torch.tensor(0.0),
            loss_recon=torch.tensor(0.0),
            loss_sample_recon=torch.tensor(0.0),
            loss_denoise=torch.tensor(0.0),
            loss_energy=torch.tensor(2.0),
            loss_recon_corr=torch.tensor(3.0),
            loss_sample_corr=torch.tensor(0.0),
            loss_recon_pca=torch.tensor(5.0),
            loss_timbre_stats=torch.tensor(0.0),
            loss_prompt_timbre_stats=torch.tensor(0.0),
            loss_fm_weight=0.0,
            lambda_recon=0.0,
            lambda_sample_recon=0.0,
            lambda_denoise=0.0,
            lambda_energy=0.5,
            lambda_recon_corr=0.0,
            lambda_sample_corr=0.0,
            lambda_recon_pca=0.0,
            lambda_timbre_stats=0.0,
            lambda_prompt_timbre_stats=0.0,
        )

        self.assertAlmostEqual(loss.item(), 1.0)

    def test_combine_training_losses_can_include_timbre_stats_loss(self):
        import torch

        loss = combine_training_losses(
            loss_fm=torch.tensor(0.0),
            loss_recon=torch.tensor(2.0),
            loss_sample_recon=torch.tensor(0.0),
            loss_denoise=torch.tensor(0.0),
            loss_energy=torch.tensor(0.0),
            loss_recon_corr=torch.tensor(0.0),
            loss_sample_corr=torch.tensor(0.0),
            loss_recon_pca=torch.tensor(0.0),
            loss_timbre_stats=torch.tensor(3.0),
            loss_prompt_timbre_stats=torch.tensor(0.0),
            loss_fm_weight=0.0,
            lambda_recon=1.0,
            lambda_sample_recon=0.0,
            lambda_denoise=0.0,
            lambda_energy=0.0,
            lambda_recon_corr=0.0,
            lambda_sample_corr=0.0,
            lambda_recon_pca=0.0,
            lambda_timbre_stats=0.5,
            lambda_prompt_timbre_stats=0.0,
        )

        self.assertAlmostEqual(loss.item(), 3.5)

    def test_combine_training_losses_can_include_prompt_timbre_stats_loss(self):
        import torch

        loss = combine_training_losses(
            loss_fm=torch.tensor(0.0),
            loss_recon=torch.tensor(2.0),
            loss_sample_recon=torch.tensor(0.0),
            loss_denoise=torch.tensor(0.0),
            loss_energy=torch.tensor(0.0),
            loss_recon_corr=torch.tensor(0.0),
            loss_sample_corr=torch.tensor(0.0),
            loss_recon_pca=torch.tensor(0.0),
            loss_timbre_stats=torch.tensor(0.0),
            loss_prompt_timbre_stats=torch.tensor(4.0),
            loss_fm_weight=0.0,
            lambda_recon=1.0,
            lambda_sample_recon=0.0,
            lambda_denoise=0.0,
            lambda_energy=0.0,
            lambda_recon_corr=0.0,
            lambda_sample_corr=0.0,
            lambda_recon_pca=0.0,
            lambda_timbre_stats=0.0,
            lambda_prompt_timbre_stats=0.25,
        )

        self.assertAlmostEqual(loss.item(), 3.0)

    def test_compose_residual_prediction_adds_baseline_and_residual(self):
        import torch

        baseline = torch.tensor([[[1.0], [2.0]]])
        residual = torch.tensor([[[0.5], [-0.25]]])

        pred = compose_residual_prediction(baseline, residual)

        self.assertTrue(torch.equal(pred, torch.tensor([[[1.5], [1.75]]])))

    def test_compose_endpoint_prediction_adds_baseline_when_residual_is_active(self):
        import torch

        raw = torch.tensor([[[0.25], [-0.5]]])
        baseline = torch.tensor([[[1.0], [2.0]]])

        pred = compose_endpoint_prediction(raw, baseline)

        self.assertTrue(torch.equal(pred, torch.tensor([[[1.25], [1.5]]])))

    def test_load_fm_head_state_can_expand_cond_projection(self):
        import torch

        model = torch.nn.Linear(5, 2)
        old_state = {
            "weight": torch.arange(6, dtype=torch.float32).reshape(2, 3),
            "bias": torch.tensor([1.0, 2.0]),
            "new_unmatched.weight": torch.ones(4),
        }

        report = load_fm_head_state(model, old_state, allow_partial=True)

        self.assertEqual(report["partial"], ["weight"])
        self.assertIn("new_unmatched.weight", report["skipped"])
        torch.testing.assert_close(model.weight[:, :3], old_state["weight"])
        torch.testing.assert_close(model.weight[:, 3:], torch.zeros(2, 2))
        torch.testing.assert_close(model.bias, old_state["bias"])

    def test_load_fm_head_state_can_insert_new_timbre_columns_before_extra_condition(self):
        import torch

        class TinyHead(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.cond_proj = torch.nn.Linear(7, 1)

        model = TinyHead()
        old_state = {
            "cond_proj.weight": torch.tensor([[10.0, 11.0, 20.0, 21.0, 30.0]]),
            "cond_proj.bias": torch.tensor([1.0]),
        }

        report = load_fm_head_state(
            model,
            old_state,
            allow_partial=True,
            cond_layout={
                "base_dim": 2,
                "old_timbre_dim": 2,
                "new_timbre_dim": 4,
                "old_extra_dim": 1,
                "new_extra_dim": 1,
            },
        )

        self.assertEqual(report["partial"], ["cond_proj.weight"])
        torch.testing.assert_close(
            model.cond_proj.weight,
            torch.tensor([[10.0, 11.0, 20.0, 21.0, 0.0, 0.0, 30.0]]),
        )
        torch.testing.assert_close(model.cond_proj.bias, old_state["cond_proj.bias"])

    def test_partial_load_keeps_old_fm_output_when_new_ctc_condition_is_zero(self):
        import torch

        from scripts.train_fm_avsr import FMHeadAVSR

        torch.manual_seed(0)
        old_model = FMHeadAVSR(n_layers=1, ctc_topk=0)
        new_model = FMHeadAVSR(
            n_layers=1,
            ctc_vocab_size=10,
            ctc_topk=2,
            ctc_token_emb_dim=3,
        )
        report = load_fm_head_state(
            new_model, old_model.state_dict(), allow_partial=True
        )
        self.assertIn("cond_proj.weight", report["partial"])

        vis = torch.randn(1, 3, 768)
        text = torch.randn(1, 3, 960)
        spk = torch.randn(1, 256)

        old_out = old_model.reconstruct_from_cond(vis, text, spk)
        new_out = new_model.reconstruct_from_cond(
            vis,
            text,
            spk,
            ctc_topk_ids=torch.zeros(1, 3, 2, dtype=torch.long),
            ctc_topk_probs=torch.zeros(1, 3, 2),
        )

        torch.testing.assert_close(new_out, old_out)

    def test_predict_energy_condition_uses_residual_baseline_when_available(self):
        import torch

        class FakeModel:
            def __init__(self, value):
                self.value = value

            def predict_extra_condition(self, *args, **kwargs):
                return torch.full((1, 2, 1), self.value)

        pred = predict_energy_condition(
            fm=FakeModel(1.0),
            residual_base=FakeModel(2.0),
            vis_down=torch.zeros(1, 2, 768),
            h_down=torch.zeros(1, 2, 960),
            spk=torch.zeros(1, 256),
        )

        self.assertTrue(torch.equal(pred, torch.full((1, 2, 1), 2.0)))

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

    def test_parse_args_loads_audio_prompt_learned_pool_from_config(self):
        old_argv = sys.argv
        try:
            with tempfile.NamedTemporaryFile("w", suffix=".yaml") as f:
                f.write("audio_prompt_learned_pool_cond: true\n")
                f.flush()
                sys.argv = [
                    "scripts/train_fm_avsr.py",
                    "--config",
                    f.name,
                ]

                args = parse_args()

            self.assertTrue(args.audio_prompt_learned_pool_cond)
        finally:
            sys.argv = old_argv

    def test_parse_args_loads_no_audio_prompt_cross_attn_from_config(self):
        old_argv = sys.argv
        try:
            with tempfile.NamedTemporaryFile("w", suffix=".yaml") as f:
                f.write("no_audio_prompt_cross_attn: true\n")
                f.flush()
                sys.argv = [
                    "scripts/train_fm_avsr.py",
                    "--config",
                    f.name,
                ]

                args = parse_args()

            self.assertTrue(args.no_audio_prompt_cross_attn)
        finally:
            sys.argv = old_argv

    def test_parse_args_loads_audio_prompt_cross_attn_pool_from_config(self):
        old_argv = sys.argv
        try:
            with tempfile.NamedTemporaryFile("w", suffix=".yaml") as f:
                f.write("audio_prompt_cross_attn_pool: true\n")
                f.flush()
                sys.argv = [
                    "scripts/train_fm_avsr.py",
                    "--config",
                    f.name,
                ]

                args = parse_args()

            self.assertTrue(args.audio_prompt_cross_attn_pool)
        finally:
            sys.argv = old_argv

    def test_parse_args_loads_audio_prompt_ref_mode_from_config(self):
        old_argv = sys.argv
        try:
            with tempfile.NamedTemporaryFile("w", suffix=".yaml") as f:
                f.write("audio_prompt_ref_mode: same_parent_next\n")
                f.flush()
                sys.argv = [
                    "scripts/train_fm_avsr.py",
                    "--config",
                    f.name,
                ]

                args = parse_args()

            self.assertEqual(args.audio_prompt_ref_mode, "same_parent_next")
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

    def test_prepare_conditions_returns_timbre_condition_when_present(self):
        import torch

        batch = {
            "enc": np.ones((1, 6, 768), dtype=np.float32),
            "latent": np.zeros((1, 3, 512), dtype=np.float32),
            "latent_lens": np.array([3], dtype=np.int64),
            "speaker": np.ones((1, 256), dtype=np.float32),
            "h_lm": None,
            "lens_L": None,
            "timbre_cond": np.arange(1024, dtype=np.float32)[None, :],
        }

        prepared = prepare_conditions(batch, "cpu")
        timbre_cond = prepared[7]

        self.assertEqual(tuple(timbre_cond.shape), (1, 1024))
        self.assertTrue(torch.allclose(timbre_cond[0, :3].float(), torch.tensor([0.0, 1.0, 2.0])))

    def test_prepare_conditions_returns_audio_prompt_when_present(self):
        batch = {
            "enc": np.ones((1, 6, 768), dtype=np.float32),
            "latent": np.zeros((1, 3, 512), dtype=np.float32),
            "latent_lens": np.array([3], dtype=np.int64),
            "speaker": np.ones((1, 256), dtype=np.float32),
            "h_lm": None,
            "lens_L": None,
            "audio_prompt": np.ones((1, 2, 512), dtype=np.float32),
        }

        prepared = prepare_conditions(batch, "cpu")
        audio_prompt = prepared[8]

        self.assertEqual(tuple(audio_prompt.shape), (1, 2, 512))

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

    def test_prepare_conditions_can_use_gt_energy_extra_condition(self):
        import torch

        batch = {
            "enc": np.ones((1, 6, 768), dtype=np.float32),
            "latent": np.zeros((1, 3, 512), dtype=np.float32),
            "latent_lens": np.array([3], dtype=np.int64),
            "speaker": np.ones((1, 256), dtype=np.float32),
            "h_lm": None,
            "lens_L": None,
            "energy": np.array([[[1.0], [2.0], [3.0]]], dtype=np.float32),
        }

        *_, extra, ctc_ids, ctc_probs = prepare_conditions(
            batch,
            "cpu",
            condition_mode="video_only",
            energy_condition_mode="gt",
        )

        self.assertEqual(tuple(extra.shape), (1, 3, 1))
        self.assertIsNone(ctc_ids)
        self.assertIsNone(ctc_probs)
        self.assertTrue(torch.allclose(extra[0, :, 0].float(), torch.tensor([1.0, 2.0, 3.0])))

    def test_predicted_energy_mode_reserves_extra_dim_without_leaking_gt_energy(self):
        batch = {
            "enc": np.ones((1, 6, 768), dtype=np.float32),
            "latent": np.zeros((1, 3, 512), dtype=np.float32),
            "latent_lens": np.array([3], dtype=np.int64),
            "speaker": np.ones((1, 256), dtype=np.float32),
            "h_lm": None,
            "lens_L": None,
            "energy": np.array([[[1.0], [2.0], [3.0]]], dtype=np.float32),
        }

        *_, extra, _, _ = prepare_conditions(
            batch,
            "cpu",
            condition_mode="video_only",
            energy_condition_mode="pred",
        )

        self.assertEqual(energy_extra_dim("pred"), 1)
        self.assertIsNone(extra)

    def test_residual_base_condition_reserves_latent_extra_dim_only_for_residuals(self):
        self.assertEqual(residual_base_extra_dim(False, None), 0)
        self.assertEqual(residual_base_extra_dim(True, None), 0)
        self.assertEqual(residual_base_extra_dim(True, "base.pt"), 512)

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
