import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from scripts.extract_timbre_cond import build_timbre_condition
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


if __name__ == "__main__":
    unittest.main()
