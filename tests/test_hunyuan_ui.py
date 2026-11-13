"""Exercise the Hunyuan subprocess boundary without loading GPU weights."""
import importlib
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
ui = importlib.import_module("run_photo_to_3d")

class GenerationTests(unittest.TestCase):
    def test_missing_photo(self):
        with self.assertRaises(ui.gr.Error):
            list(ui.generate(None))

    def test_export_survives_temporary_cleanup_and_passes_options(self):
        created = []
        payload = b"glTF" + bytes(20)
        def run(command, **kwargs):
            self.assertIn("--mirror-back", command)
            self.assertTrue(kwargs["check"])
            self.assertEqual(kwargs["timeout"], 1800)
            target = Path(command[3])
            target.write_bytes(payload)
            created.append(target)
        with patch.object(ui.subprocess, "run", side_effect=run):
            results = list(ui.generate("photo.png", True))
        cached = Path(results[-1][0])
        self.assertEqual(cached.read_bytes(), payload)
        self.assertEqual(results[-1][0], results[-1][1])
        self.assertFalse(created[0].exists())

    def test_process_failure_and_missing_output(self):
        for error in (subprocess.TimeoutExpired("model", 1800), None):
            with self.subTest(error=error):
                with patch.object(ui.subprocess, "run", side_effect=error):
                    with self.assertLogs(level="ERROR"):
                        results = list(ui.generate("photo.png"))
                self.assertEqual(results[-1][:2], (None, None))

if __name__ == "__main__":
    unittest.main()
