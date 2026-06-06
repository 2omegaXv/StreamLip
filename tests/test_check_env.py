import argparse
import tempfile
import unittest
from pathlib import Path

from scripts import check_env


class CheckEnvTest(unittest.TestCase):
    def test_required_checkpoint_paths_are_relative_to_ckpt_bundle(self):
        paths = check_env.required_checkpoint_paths()

        self.assertIn(Path("ckpt/mimi/config.json"), paths)
        self.assertIn(Path("ckpt/v5/streamlip_v5_olmo_step_002000_infer.pt"), paths)
        self.assertIn(Path("ckpt/recon/streamlip_recon_timbrefix_step_002000.pt"), paths)
        self.assertTrue(all(not path.is_absolute() for path in paths))

    def test_missing_checkpoint_check_reports_each_missing_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            missing = check_env.missing_checkpoint_paths(repo)

        self.assertIn(Path("ckpt/mimi/config.json"), missing)
        self.assertIn(Path("ckpt/auto-avsr/vsr_trlrs2lrs3vox2avsp_base.pth"), missing)

    def test_skip_imports_avoids_import_checks(self):
        args = argparse.Namespace(skip_imports=True)

        self.assertEqual(check_env.collect_import_status(args), [])

    def test_package_script_exists(self):
        script = Path("scripts/package_ckpt.sh")

        self.assertTrue(script.exists())
        self.assertIn("streamlip_ckpt_", script.read_text())


if __name__ == "__main__":
    unittest.main()
