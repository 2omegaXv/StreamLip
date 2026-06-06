import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from scripts.extract_timbre_cond import build_mel_stats, build_timbre_condition
from scripts.train_fm_avsr import FMHeadAVSR


class TimbreConditionTest(unittest.TestCase):
    def test_build_timbre_condition_uses_prefix_mean_and_std(self):
        latent = np.arange(5 * 512, dtype=np.float32).reshape(5, 512)

        cond = build_timbre_condition(latent, prompt_frames=3)

        expected_prefix = latent[:3]
        self.assertEqual(cond.shape, (1024,))
        np.testing.assert_allclose(cond[:512], expected_prefix.mean(axis=0), rtol=1e-6)
        np.testing.assert_allclose(cond[512:], expected_prefix.std(axis=0), rtol=1e-6)

    def test_build_timbre_condition_uses_all_frames_for_short_clip(self):
        latent = np.ones((2, 512), dtype=np.float32)

        cond = build_timbre_condition(latent, prompt_frames=10)

        self.assertEqual(cond.shape, (1024,))
        np.testing.assert_allclose(cond[:512], 1.0)
        np.testing.assert_allclose(cond[512:], 0.0)

    def test_build_timbre_condition_can_append_log_mel_stats(self):
        latent = np.ones((2, 512), dtype=np.float32)
        mel_stats = np.linspace(-1.0, 1.0, 160, dtype=np.float32)

        cond = build_timbre_condition(latent, prompt_frames=2, extra_stats=mel_stats)

        self.assertEqual(cond.shape, (1184,))
        np.testing.assert_allclose(cond[:512], 1.0)
        np.testing.assert_allclose(cond[512:1024], 0.0)
        np.testing.assert_allclose(cond[1024:], mel_stats)

    def test_build_mel_stats_returns_mean_and_std_for_prompt_audio(self):
        sr = 16000
        audio = np.sin(np.linspace(0.0, 32.0 * np.pi, sr, dtype=np.float32))

        stats = build_mel_stats(audio, sr, prompt_seconds=1.0, n_mels=40)

        self.assertEqual(stats.shape, (80,))
        self.assertTrue(np.isfinite(stats).all())
        self.assertGreater(float(stats[40:].max()), 0.0)

    def test_fm_head_accepts_timbre_condition_and_keeps_output_shape(self):
        fm = FMHeadAVSR(n_layers=1, timbre_condition_dim=1024)
        v = torch.zeros(2, 4, 768)
        h = torch.zeros(2, 4, 960)
        spk = torch.zeros(2, 256)
        timbre = torch.ones(2, 1024)

        out = fm.reconstruct_from_cond(v, h, spk, timbre_cond=timbre)

        self.assertEqual(tuple(out.shape), (2, 4, 512))

    def test_fm_head_accepts_audio_prompt_tokens_and_keeps_output_shape(self):
        fm = FMHeadAVSR(n_layers=1, use_cross_attn=True, audio_prompt_dim=512)
        v = torch.zeros(2, 4, 768)
        h = torch.zeros(2, 4, 960)
        spk = torch.zeros(2, 256)
        prompt = torch.ones(2, 3, 512)

        out = fm.reconstruct_from_cond(v, h, spk, audio_prompt=prompt)

        self.assertEqual(tuple(out.shape), (2, 4, 512))

    def test_audio_prompt_cross_attention_ignores_text_mask_without_text_token_cross_attention(self):
        fm = FMHeadAVSR(n_layers=1, use_cross_attn=True, audio_prompt_dim=512)
        v = torch.zeros(2, 4, 768)
        h = torch.zeros(2, 4, 960)
        spk = torch.zeros(2, 256)
        text_tokens = torch.zeros(2, 5, 960)
        text_mask = torch.ones(2, 5, dtype=torch.bool)
        prompt = torch.ones(2, 3, 512)

        out = fm.reconstruct_from_cond(
            v,
            h,
            spk,
            text_tokens=text_tokens,
            text_token_mask=text_mask,
            audio_prompt=prompt,
        )

        self.assertEqual(tuple(out.shape), (2, 4, 512))

    def test_audio_prompt_pool_condition_affects_frame_condition_without_cross_attention(self):
        fm = FMHeadAVSR(
            n_layers=1,
            use_cross_attn=False,
            audio_prompt_dim=512,
            audio_prompt_pool_cond=True,
        )
        v = torch.zeros(2, 4, 768)
        h = torch.zeros(2, 4, 960)
        spk = torch.zeros(2, 256)
        prompt_a = torch.zeros(2, 3, 512)
        prompt_b = torch.ones(2, 3, 512)

        cond_a, _ = fm._build_cond(v, h, spk, audio_prompt=prompt_a)
        cond_b, _ = fm._build_cond(v, h, spk, audio_prompt=prompt_b)

        self.assertGreater(float((cond_a - cond_b).detach().abs().sum()), 0.0)

    def test_audio_prompt_tokens_can_be_excluded_from_cross_attention(self):
        fm = FMHeadAVSR(
            n_layers=1,
            use_cross_attn=True,
            audio_prompt_dim=512,
            audio_prompt_pool_cond=True,
            audio_prompt_cross_attn=False,
        )
        v = torch.zeros(2, 4, 768)
        h = torch.zeros(2, 4, 960)
        spk = torch.zeros(2, 256)
        prompt = torch.ones(2, 3, 512)

        built = fm._build_cond(v, h, spk, audio_prompt=prompt)

        self.assertFalse(isinstance(built, tuple))
        self.assertEqual(tuple(built.shape), (2, 4, 512))

    def test_audio_prompt_cross_attention_can_use_pooled_token_only(self):
        fm = FMHeadAVSR(
            n_layers=1,
            use_cross_attn=True,
            audio_prompt_dim=512,
            audio_prompt_cross_attn_pool=True,
        )
        v = torch.zeros(2, 4, 768)
        h = torch.zeros(2, 4, 960)
        spk = torch.zeros(2, 256)
        prompt = torch.randn(2, 3, 512)

        cond, cond_tokens = fm._build_cond(v, h, spk, audio_prompt=prompt)

        expected = fm.audio_prompt_proj(prompt).mean(dim=1, keepdim=True)
        self.assertEqual(tuple(cond.shape), (2, 4, 512))
        self.assertEqual(tuple(cond_tokens.shape), (2, 1, 512))
        torch.testing.assert_close(cond_tokens, expected)

    def test_audio_prompt_cross_attention_can_use_multiple_pooled_tokens(self):
        fm = FMHeadAVSR(
            n_layers=1,
            use_cross_attn=True,
            audio_prompt_dim=512,
            audio_prompt_cross_attn_pool_tokens=4,
        )
        v = torch.zeros(2, 4, 768)
        h = torch.zeros(2, 4, 960)
        spk = torch.zeros(2, 256)
        prompt = torch.randn(2, 8, 512)

        cond, cond_tokens = fm._build_cond(v, h, spk, audio_prompt=prompt)

        projected = fm.audio_prompt_proj(prompt)
        expected = projected.reshape(2, 4, 2, 512).mean(dim=2)
        self.assertEqual(tuple(cond.shape), (2, 4, 512))
        self.assertEqual(tuple(cond_tokens.shape), (2, 4, 512))
        torch.testing.assert_close(cond_tokens, expected)

    def test_audio_prompt_stat_pool_condition_starts_as_noop(self):
        torch.manual_seed(0)
        mean_pool = FMHeadAVSR(
            n_layers=1,
            use_cross_attn=False,
            audio_prompt_dim=512,
            audio_prompt_pool_cond=True,
        )
        stat_pool = FMHeadAVSR(
            n_layers=1,
            use_cross_attn=False,
            audio_prompt_dim=512,
            audio_prompt_pool_cond=True,
            audio_prompt_stat_pool_cond=True,
        )
        stat_pool.load_state_dict(mean_pool.state_dict(), strict=False)
        v = torch.zeros(2, 4, 768)
        h = torch.zeros(2, 4, 960)
        spk = torch.zeros(2, 256)
        prompt = torch.randn(2, 3, 512)

        cond_mean, _ = mean_pool._build_cond(v, h, spk, audio_prompt=prompt)
        cond_stat, _ = stat_pool._build_cond(v, h, spk, audio_prompt=prompt)

        torch.testing.assert_close(cond_stat, cond_mean)


if __name__ == "__main__":
    unittest.main()
