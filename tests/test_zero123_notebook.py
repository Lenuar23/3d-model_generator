"""Offline checks for notebook packaging, conditioning and batch reporting."""

from contextlib import nullcontext, redirect_stderr, redirect_stdout
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

try:
    from PIL import Image
except ImportError:
    Image = None


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts/run_zero123.py"
spec = importlib.util.spec_from_file_location("run_zero123", RUNNER)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class NotebookTests(unittest.TestCase):
    def test_unsupported_colab_python_stops_before_starting_gpu_check(self):
        notebook = json.loads((ROOT / "notebooks/zero123plus.ipynb").read_text(encoding="utf-8"))
        config = next(cell for cell in notebook["cells"] if cell["id"] == "configure")
        with (patch.object(sys, "version_info", (3, 13, 18)),
              patch.object(subprocess, "run") as run):
            with self.assertRaisesRegex(RuntimeError, r"Runtime Version.*2026\.07"):
                exec(compile("".join(config["source"]), "zero123plus.ipynb:configure", "exec"), {})
        run.assert_not_called()

    def test_embedded_runtime_matches_local_script_and_requirements(self):
        notebook_path = ROOT / "notebooks/zero123plus.ipynb"
        notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
        self.assertEqual(notebook["nbformat"], 4)
        self.assertEqual(notebook["metadata"]["accelerator"], "GPU")
        code = {}
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                self.assertEqual(cell["outputs"], [])
                self.assertIsNone(cell["execution_count"])
                code[cell["id"]] = "".join(cell["source"])
        self.assertEqual(code.pop("runner-source").split("\n", 1)[1], RUNNER.read_text(encoding="utf-8"))
        self.assertEqual(code.pop("requirements").split("\n", 1)[1],
                         (ROOT / "notebooks/zero123_requirements.in").read_text(encoding="utf-8"))
        for name, source in code.items():
            compile(source, f"zero123plus.ipynb:{name}", "exec")

    def test_batch_checks_exact_count_and_content(self):
        with tempfile.TemporaryDirectory() as folder:
            photos_dir = Path(folder)
            manifest = []
            for index in range(10):
                filename = f"photo_{index}.jpg"
                payload = bytes([index])
                (photos_dir / filename).write_bytes(payload)
                manifest.append(dict(filename=filename, category="simple", sha256=hashlib.sha256(payload).hexdigest()))

            def open_manifest(*args, **kwargs):
                return io.BytesIO(json.dumps(dict(photos=manifest)).encode())

            with patch.object(runner, "urlopen", side_effect=open_manifest):
                self.assertEqual(len(runner.load_batch(photos_dir)), 10)
                (photos_dir / "photo_0.jpg").write_bytes(b"different")
                with self.assertRaisesRegex(ValueError, "Missing or changed test photograph"):
                    runner.load_batch(photos_dir)
                manifest.pop()
                with self.assertRaisesRegex(ValueError, "ten distinct"):
                    runner.load_batch(photos_dir)


@unittest.skipIf(Image is None, "Pillow is available in the isolated Zero123++ runtime")
class RunnerTests(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.folder = Path(folder.name)

    def test_grayscale_conditioning_keeps_aspect_and_uses_rgb(self):
        path = self.folder / "gray.png"
        Image.new("L", (20, 40), 50).save(path)

        def segment(photo):
            self.assertEqual(photo.mode, "RGB")
            rgba = photo.convert("RGBA")
            alpha = Image.new("L", photo.size)
            alpha.paste(255, (5, 10, 15, 30))
            rgba.putalpha(alpha)
            return rgba

        condition, method = runner.condition_photo(path, segment)
        self.assertEqual((condition.mode, condition.size, method), ("RGB", (320, 320), "rembg_u2netp"))
        self.assertEqual(condition.getpixel((96, 32)), (50, 50, 50))
        self.assertEqual(condition.getpixel((95, 32)), runner.GRAY)
        self.assertEqual(condition.getpixel((160, 31)), runner.GRAY)

    def test_existing_alpha_skips_segmentation_and_empty_mask_falls_back(self):
        path = self.folder / "alpha.png"
        photo = Image.new("RGBA", (20, 20))
        photo.paste((255, 0, 0, 255), (5, 5, 15, 15))
        photo.save(path)
        segment = Mock()
        condition, method = runner.condition_photo(path, segment)
        self.assertEqual(method, "source_alpha")
        self.assertEqual(condition.getpixel((160, 160)), (255, 0, 0))
        segment.assert_not_called()

        photo.convert("RGB").save(path)
        segment.return_value = Image.new("RGBA", photo.size)
        condition, method = runner.condition_photo(path, segment)
        self.assertEqual(method, "original_rgb_empty_mask")
        self.assertEqual(condition.getpixel((160, 160)), (255, 0, 0))

    def test_generation_uses_seed_and_checks_grid_dimensions(self):
        photo = Image.new("RGB", (320, 320))
        grid = Image.new("RGB", runner.GRID_SIZE)
        grid.putpixel((5, 5), (255, 0, 0))
        pipe = Mock(return_value=SimpleNamespace(images=[grid]))
        torch = Mock()
        torch.inference_mode.side_effect = nullcontext
        output = runner.generate_grid(pipe, torch, photo, 42, 28)
        self.assertEqual(output.size, runner.GRID_SIZE)
        torch.manual_seed.assert_called_once_with(42)
        torch.Generator.return_value.manual_seed.assert_called_once_with(42)
        pipe.assert_called_once_with(photo, num_inference_steps=28, guidance_scale=4.0,
                                     generator=torch.Generator.return_value.manual_seed.return_value)
        pipe.return_value.images = [Image.new("RGB", (640, 640))]
        with self.assertRaisesRegex(ValueError, "Expected grid"):
            runner.generate_grid(pipe, torch, photo, 42, 28)
        pipe.return_value.images = [Image.new("RGB", runner.GRID_SIZE)]
        with self.assertRaisesRegex(ValueError, "uniform grid"):
            runner.generate_grid(pipe, torch, photo, 42, 28)

    def test_failed_generation_preserves_successful_grid_and_reports_failure(self):
        inputs = [(self.folder / name, "simple", "digest") for name in ("one.jpg", "two.jpg")]
        output = self.folder / "output"
        torch = Mock()
        torch.cuda.mem_get_info.return_value = (4, 6)
        torch.cuda.get_device_name.return_value = "test GPU"
        torch.cuda.max_memory_allocated.return_value = 1
        torch.version.cuda = "test CUDA"
        grid = Image.new("RGB", runner.GRID_SIZE)

        def prepare(*args, **kwargs):
            report_path = output / "run.json"
            report = json.loads(report_path.read_text(encoding="utf-8"))
            for index, photo in enumerate(report["photos"]):
                condition = f"{index}-condition.png"
                Image.new("RGB", (320, 320)).save(output / condition)
                photo.update(status="prepared", condition=condition)
            runner.write_json(report_path, report)

        with (redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()),
              patch.dict(sys.modules, {"torch": torch}),
              patch.object(runner.subprocess, "run", side_effect=prepare),
              patch.object(runner.importlib.metadata, "version", return_value="test"),
              patch.object(runner, "load_pipeline"),
              patch.object(runner, "generate_grid", side_effect=[grid, RuntimeError("test failure")])):
            self.assertFalse(runner.run(inputs, output, seed=42, steps=28, vae_tiling=False, batch=True))
        report = json.loads((output / "run.json").read_text(encoding="utf-8"))
        self.assertEqual((report["status"], report["succeeded"]), ("failed", 1))
        self.assertTrue((output / report["photos"][0]["grid"]).is_file())
        self.assertEqual(report["photos"][1]["stage"], "inference")
        self.assertIn("test failure", report["photos"][1]["error"])

    def test_preprocessing_process_failure_preserves_progress(self):
        inputs = [(self.folder / name, "simple", "digest") for name in ("one.jpg", "two.jpg")]
        output = self.folder / "output"
        report_path = output / "run.json"

        def partial_preparation(command, **kwargs):
            report = json.loads(report_path.read_text(encoding="utf-8"))
            report["photos"][0].update(status="prepared", condition="one-condition.png")
            runner.write_json(report_path, report)
            raise subprocess.CalledProcessError(17, command)

        with (patch.object(runner.subprocess, "run", side_effect=partial_preparation),
              patch.object(runner, "load_pipeline") as load_pipeline):
            with self.assertRaises(subprocess.CalledProcessError):
                runner.run(inputs, output, seed=42, steps=28, vae_tiling=False, batch=True)
        load_pipeline.assert_not_called()
        report = json.loads(report_path.read_text(encoding="utf-8"))
        self.assertEqual(report["status"], "failed")
        self.assertIn("17", report["fatal_error"])
        self.assertEqual([item["status"] for item in report["photos"]], ["prepared", "pending"])
        self.assertEqual(report["photos"][0]["condition"], "one-condition.png")


if __name__ == "__main__":
    unittest.main()
