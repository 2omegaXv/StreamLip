import unittest

import numpy as np

from scripts.run_raw_video_avsr_recon_pipeline import (
    build_audio_prompt_condition,
    build_timbre_condition_for_pipeline,
    recon_wav_start_frame,
    result_video_names,
)


class RawVideoPipelineTest(unittest.TestCase):
    def test_build_audio_prompt_condition_pads_short_ref_latent(self):
        latent = np.arange(3 * 512, dtype=np.float32).reshape(3, 512)

        prompt = build_audio_prompt_condition(latent, prompt_frames=5)

        self.assertEqual(prompt.shape, (5, 512))
        np.testing.assert_allclose(prompt[:3], latent)
        np.testing.assert_allclose(prompt[3:], 0.0)

    def test_build_audio_prompt_condition_can_make_default_zero_prompt(self):
        prompt = build_audio_prompt_condition(None, prompt_frames=4)

        self.assertEqual(prompt.shape, (4, 512))
        np.testing.assert_allclose(prompt, 0.0)

    def test_build_timbre_condition_can_make_default_zero_timbre(self):
        timbre = build_timbre_condition_for_pipeline(None)

        self.assertEqual(timbre.shape, (1024,))
        np.testing.assert_allclose(timbre, 0.0)

    def test_silent_mode_exports_post_prompt_output_names(self):
        pred_name, gt_name = result_video_names("demo", silent_input=True)

        self.assertEqual(pred_name, "demo_pred_post3s.mp4")
        self.assertIsNone(gt_name)
        self.assertEqual(recon_wav_start_frame(silent_input=True), 38)

    def test_audio_prompt_mode_keeps_existing_post_prompt_names(self):
        pred_name, gt_name = result_video_names("demo", silent_input=False)

        self.assertEqual(pred_name, "demo_pred_prompt3s_post3s.mp4")
        self.assertEqual(gt_name, "demo_gt_mimi_post3s.mp4")
        self.assertEqual(recon_wav_start_frame(silent_input=False), 38)


if __name__ == "__main__":
    unittest.main()
