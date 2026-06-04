import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.build_pocket_tts_teacher_cache import build_teacher_cache, read_clip_text


class PocketTTSTeacherCacheTest(unittest.TestCase):
    def test_read_clip_text_prefers_text_json_words(self):
        with tempfile.TemporaryDirectory() as tmp:
            clip = Path(tmp)
            (clip / "avsr_text.txt").write_text("")
            (clip / "text.json").write_text(
                json.dumps({"words": [{"word": "HELLO"}, {"word": "WORLD"}]})
            )

            self.assertEqual(read_clip_text(clip), "HELLO WORLD")

    def test_build_teacher_cache_writes_manifest_and_skips_existing_items(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            clip = root / "pretrain" / "video_a" / "00001"
            clip.mkdir(parents=True)
            (clip / "avsr_text.txt").write_text("Hello teacher cache.\n")
            clip_list = root / "clips.txt"
            clip_list.write_text(str(clip) + "\n")
            cache_root = root / "teacher_cache"

            calls = {"tts": 0, "mimi": 0}

            def fake_tts(text, voice, out_wav):
                calls["tts"] += 1
                self.assertEqual(text, "Hello teacher cache.")
                self.assertEqual(voice, "alba")
                sf.write(out_wav, np.zeros(2400, dtype=np.float32), 24000)
                return 24000, 2400

            def fake_mimi(wav_path, out_npz):
                calls["mimi"] += 1
                np.savez(out_npz, codes=np.zeros((1, 32, 2), dtype=np.int64))
                return (1, 32, 2)

            first = build_teacher_cache(
                clip_list=clip_list,
                cache_root=cache_root,
                n=1,
                voice="alba",
                tts_fn=fake_tts,
                mimi_fn=fake_mimi,
            )
            second = build_teacher_cache(
                clip_list=clip_list,
                cache_root=cache_root,
                n=1,
                voice="alba",
                tts_fn=fake_tts,
                mimi_fn=fake_mimi,
            )

            self.assertEqual(calls, {"tts": 1, "mimi": 1})
            self.assertEqual(len(first), 1)
            self.assertEqual(len(second), 1)
            self.assertTrue(first[0]["generated"])
            self.assertFalse(second[0]["generated"])
            self.assertEqual(first[0]["codes_shape"], [1, 32, 2])

            manifest = cache_root / "manifest.jsonl"
            rows = [json.loads(line) for line in manifest.read_text().splitlines()]
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["text"], "Hello teacher cache.")
            self.assertTrue(Path(rows[0]["teacher_wav"]).exists())
            self.assertTrue(Path(rows[0]["mimi_codes"]).exists())

    def test_build_teacher_cache_resolves_relative_clips_against_data_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            clip = root / "processed" / "pretrain" / "video_b" / "00002"
            clip.mkdir(parents=True)
            (clip / "avsr_text.txt").write_text("Relative path clip.\n")
            clip_list = root / "clips.txt"
            clip_list.write_text("pretrain/video_b/00002\n")

            def fake_tts(text, voice, out_wav):
                sf.write(out_wav, np.zeros(1200, dtype=np.float32), 24000)
                return 24000, 1200

            def fake_mimi(wav_path, out_npz):
                np.savez(out_npz, codes=np.zeros((1, 32, 1), dtype=np.int64))
                return (1, 32, 1)

            rows = build_teacher_cache(
                clip_list=clip_list,
                cache_root=root / "teacher_cache",
                data_root=root / "processed",
                n=1,
                voice="alba",
                tts_fn=fake_tts,
                mimi_fn=fake_mimi,
            )

            self.assertEqual(rows[0]["clip"], str(clip))
            self.assertEqual(rows[0]["text"], "Relative path clip.")


if __name__ == "__main__":
    unittest.main()
