import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from streaminlip.fm_avsr_dataset import validate_latent_frame_rate


class FMAVSRDatasetTest(unittest.TestCase):
    def test_reject_25hz_latent(self):
        lat = np.arange(10 * 2, dtype=np.float32).reshape(10, 2)
        with self.assertRaisesRegex(ValueError, "25Hz"):
            validate_latent_frame_rate(lat, enc_len=10, clip="dummy")

    def test_keep_existing_12p5hz_latent(self):
        lat = np.arange(5 * 2, dtype=np.float32).reshape(5, 2)
        validated = validate_latent_frame_rate(lat, enc_len=10, clip="dummy")
        np.testing.assert_array_equal(validated, lat)


if __name__ == "__main__":
    unittest.main()
