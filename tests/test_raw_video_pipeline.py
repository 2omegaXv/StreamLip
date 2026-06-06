import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from scripts.run_raw_video_avsr_recon_pipeline import (
    build_audio_prompt_condition,
    build_timbre_condition_for_pipeline,
    build_eval_recon_command,
    build_extract_smollm2_command,
    build_extract_v5_text_command,
    effective_silent_input,
    should_concat_ref_prompt_prefix,
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

    def test_silent_ref_mode_uses_black_video_ref_audio_prefix(self):
        self.assertTrue(should_concat_ref_prompt_prefix(silent_input=True, has_ref_audio=True))
        self.assertFalse(should_concat_ref_prompt_prefix(silent_input=True, has_ref_audio=False))
        self.assertFalse(should_concat_ref_prompt_prefix(silent_input=False, has_ref_audio=True))

    def test_silent_mode_is_auto_enabled_for_video_without_audio(self):
        with mock.patch(
            "scripts.run_raw_video_avsr_recon_pipeline.has_audio_stream",
            return_value=False,
        ):
            self.assertTrue(effective_silent_input(False, Path("silent.mp4")))

    def test_silent_mode_respects_explicit_user_request(self):
        with mock.patch(
            "scripts.run_raw_video_avsr_recon_pipeline.has_audio_stream",
            return_value=True,
        ):
            self.assertTrue(effective_silent_input(True, Path("with_audio.mp4")))

    def test_audio_video_keeps_audio_mode_by_default(self):
        with mock.patch(
            "scripts.run_raw_video_avsr_recon_pipeline.has_audio_stream",
            return_value=True,
        ):
            self.assertFalse(effective_silent_input(False, Path("with_audio.mp4")))

    def test_default_text_pipeline_uses_streamlip_v5(self):
        cmd = build_extract_v5_text_command(
            processed_root="processed",
            clip_list="clips.txt",
            avsr_ckpt="auto_avsr.pt",
            v5_ckpt="v5.pt",
            v5_lm_path="olmo",
        )

        self.assertIn("extract_v5_text.py", cmd[1])
        self.assertIn("--input_name", cmd)
        self.assertEqual(cmd[cmd.index("--input_name") + 1], "avsr_enc_lipavsr.npy")
        self.assertIn("--output_name", cmd)
        self.assertEqual(cmd[cmd.index("--output_name") + 1], "streamlip_v5_text.txt")

    def test_default_hidden_and_recon_commands_read_v5_text_source(self):
        smollm_cmd = build_extract_smollm2_command(
            processed_root="processed",
            clip_list="clips.txt",
            smollm2_path="smollm2",
            text_source="v5",
        )
        recon_cmd = build_eval_recon_command(
            processed_root="processed",
            clip_list="clips.txt",
            output_dir="out",
            config="config.yaml",
            ckpt="model.pt",
            text_source="v5",
            silent_input=False,
        )

        self.assertEqual(smollm_cmd[smollm_cmd.index("--text_source") + 1], "v5")
        self.assertEqual(recon_cmd[recon_cmd.index("--text_source") + 1], "v5")


if __name__ == "__main__":
    unittest.main()
