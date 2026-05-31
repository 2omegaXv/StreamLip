import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from streaminlip.fm_avsr_dataset import validate_latent_frame_rate
from scripts.train_fm_avsr import crop_batch_to_latent_window


class FMAVSRDatasetTest(unittest.TestCase):
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
        enc_crop, latent_crop = crop_batch_to_latent_window(
            enc, latent, crop_ta=4, rng=rng
        )

        self.assertEqual(latent_crop.shape, (2, 4, 1))
        self.assertEqual(enc_crop.shape, (2, 8, 1))
        for b in range(2):
            start = int(latent_crop[b, 0, 0] - latent[b, 0, 0])
            self.assertEqual(start % 10, start)
            np.testing.assert_array_equal(
                enc_crop[b, :, 0],
                enc[b, start * 2 : start * 2 + 8, 0],
            )


if __name__ == "__main__":
    unittest.main()
