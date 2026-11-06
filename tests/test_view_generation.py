"""Offline tests for Zero123++ grid framing, storage and subprocess boundaries."""

import importlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
generator = importlib.import_module("3d_model_soft.infrastructure.ai.zero123_generator")
storage = importlib.import_module("3d_model_soft.infrastructure.storage.views_repository")
synthetic = importlib.import_module("3d_model_soft.infrastructure.synthetic_dataset")
synthetic_service = importlib.import_module("3d_model_soft.services.synthetic")
synthetic_cameras = synthetic_service.zero123_cameras
carving = importlib.import_module("3d_model_soft.services.carving")
ViewGenerator = importlib.import_module("3d_model_soft.domain.interfaces").ViewGenerator
models = importlib.import_module("3d_model_soft.domain.models")
Camera = models.Camera


def make_grid():
    grid = np.empty((960, 640, 3), dtype=np.uint8)
    for index in range(6):
        row, column = divmod(index, 2)
        y, x = row * 320, column * 320
        grid[y:y + 320, x:x + 320] = [index * 29, index * 13, index * 17]
        grid[y + 19, x + 53] = [255, 128, 64]
    return grid


def soft_alpha(tile):
    alpha = np.zeros((*tile.shape[:2], 1), dtype=np.uint8)
    alpha[19, 52:56, 0] = [0, 127, 128, 255]
    return np.concatenate((tile, alpha), axis=2)


class GenerationTests(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.folder = Path(folder.name)
        self.source = self.folder / "photo.jpg"
        self.source.touch()
        self.output = self.folder / "runs"
        self.grid = make_grid()
        self.backend = Mock(return_value=self.grid)
        self.remover = Mock()
        self.remover.remove.side_effect = soft_alpha
        self.generator = generator.Zero123ViewGenerator(self.backend, self.remover, self.output, distance=3.5)

    def test_grid_order_camera_values_and_soft_alpha_preserve_framing(self):
        before = self.grid.copy()

        def segment(tile):
            rgba = soft_alpha(tile)
            tile[:] = 255  # A backend that mutates its input must not modify the grid.
            return rgba

        self.remover.remove.side_effect = segment
        target = self.generator.generate_views(self.source, 42)
        self.assertIsInstance(self.generator, ViewGenerator)
        self.backend.assert_called_once_with(self.source.resolve(), 42)
        self.assertEqual(self.remover.remove.call_count, 6)
        np.testing.assert_array_equal(self.grid, before)
        self.assertEqual(target.name, "views")
        self.assertEqual(target.parent.parent, self.output)
        self.assertEqual(len(list(target.iterdir())), 7)

        names = [f"view_{index:02d}.png" for index in range(6)]
        metadata = json.loads((target / "cameras.json").read_text(encoding="utf-8"))
        views = storage.ViewsRepository().load(target)
        self.assertEqual(tuple(view.camera for view in views), synthetic_cameras("v1.2", size=320, distance=3.5))
        for index, (name, view) in enumerate(zip(names, views, strict=True)):
            row, column = divmod(index, 2)
            y, x = row * 320, column * 320
            with self.subTest(name=name), Image.open(target / name) as image:
                self.assertEqual((image.format, image.mode, image.size), ("PNG", "RGBA", (320, 320)))
                np.testing.assert_array_equal(np.array(image), soft_alpha(before[y:y + 320, x:x + 320]))
            np.testing.assert_array_equal(view.image, before[y:y + 320, x:x + 320])
            np.testing.assert_array_equal(view.mask[19, 52:56], [False, False, True, True])
            self.assertEqual(metadata[name], dict(azimuth=30 + 60 * index, elevation=(20, -10)[index % 2],
                                                  distance=3.5, fov=30.0))
        strict_views = storage.ViewsRepository().load(target, alpha_threshold=255)
        self.assertEqual(np.count_nonzero(strict_views[0].mask), 1)

    def test_invalid_seed_or_grid_never_reaches_background_removal(self):
        for seed in (True, -1, 2**63, 1.25):
            with self.subTest(seed=seed), self.assertRaisesRegex(ValueError, "seed"):
                self.generator.generate_views(self.source, seed)
        self.backend.assert_not_called()
        self.backend.return_value = self.grid.transpose(1, 0, 2)
        with self.assertRaisesRegex(ValueError, "640 x 960"):
            self.generator.generate_views(self.source, 42)
        self.remover.remove.assert_not_called()
        self.assertFalse(self.output.exists())

    def test_masking_or_invalid_rgba_leaves_no_published_views(self):
        good = soft_alpha(self.grid[:320, :320])
        self.remover.remove.side_effect = [good, RuntimeError("segmentation failed")]
        with self.assertRaisesRegex(RuntimeError, "segmentation failed"):
            self.generator.generate_views(self.source, 42)
        self.assertFalse(self.output.exists())

        self.remover.remove.side_effect = lambda tile: tile
        with self.assertRaisesRegex(ValueError, "uint8 RGBA"):
            self.generator.generate_views(self.source, 42)
        self.assertEqual(list(self.output.iterdir()), [])

    def test_repeat_generation_keeps_earlier_bundle(self):
        first = self.generator.generate_views(self.source, 42)
        second = self.generator.generate_views(self.source, 43)
        self.assertNotEqual(first, second)
        self.assertTrue((first / "cameras.json").is_file())
        self.assertTrue((second / "cameras.json").is_file())


class RepositoryTests(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.folder = Path(folder.name)
        self.target = self.folder / "views"
        self.repository = storage.ViewsRepository()
        self.camera = Camera(30, 20, 4, 30, 16, 12)
        self.rgba = np.zeros((12, 16, 4), dtype=np.uint8)

    def test_existing_synthetic_fixture_round_trips_into_carving(self):
        radii = (0.6, 0.8, 0.5)
        output = synthetic.generate_dataset(self.folder / "fixture", radii=radii, size=64, resolution=24)
        views = self.repository.load(output / "views")
        self.assertEqual(tuple(view.camera for view in views), synthetic_cameras(size=64))
        for view in views:
            np.testing.assert_array_equal(view.mask, synthetic_service.ellipsoid_mask(view.camera, radii))
        with np.load(output / "ground_truth.npz") as fixture:
            truth = fixture["occupancy"]
            spacing = (float(fixture["spacing"]),) * 3
            origin = tuple(fixture[axis][0] - step / 2 for axis, step in zip("xyz", spacing, strict=True))
        grid = models.VoxelGrid(np.ones_like(truth), origin, spacing)
        carved = carving.carve(views, grid).occupancy
        intersection = np.count_nonzero(carved & truth)
        union = np.count_nonzero(carved | truth)
        self.assertGreater(intersection / union, 0.8)

    def test_save_failure_never_publishes_partial_bundle(self):
        views = {name: (self.camera, self.rgba) for name in ("view_00.png", "view_01.png")}
        save = Image.Image.save

        def fail_on_second(image, path, **kwargs):
            if path.name == "view_01.png":
                raise OSError("test write failure")
            save(image, path, **kwargs)

        with patch.object(Image.Image, "save", new=fail_on_second):
            with self.assertRaisesRegex(OSError, "test write failure"):
                self.repository.save(self.target, views)
        self.assertEqual(list(self.folder.iterdir()), [])

    def test_save_refuses_to_replace_existing_output(self):
        self.repository.save(self.target, {"view_00.png": (self.camera, self.rgba)})
        before = (self.target / "view_00.png").read_bytes()
        with self.assertRaises(FileExistsError):
            self.repository.save(self.target, {"view_00.png": (self.camera, 255 - self.rgba)})
        self.assertEqual((self.target / "view_00.png").read_bytes(), before)

    def test_read_rejects_invalid_manifest_and_camera_values(self):
        path = self.repository.save(self.target, {"view_00.png": (self.camera, self.rgba)}) / "cameras.json"
        valid = dict(azimuth=30, elevation=20, distance=4, fov=30)
        cases = (
            (json.dumps({"../outside.png": valid}), "filename"),
            (json.dumps({"C:\\outside.png": valid}), "filename"),
            ('{"view_00.png": {}, "view_00.png": {}}', "Duplicate JSON key"),
            (json.dumps({"view_00.png": dict(valid, azimuth=True)}), "camera fields"),
            (json.dumps({"view_00.png": dict(valid, distance=float("inf"))}), "camera"),
            (json.dumps({"view_00.png": dict(valid, fov=180)}), "camera"),
            (json.dumps({"view_01.png": valid}), "must match"),
        )
        for manifest, message in cases:
            with self.subTest(manifest=manifest):
                path.write_text(manifest, encoding="utf-8")
                with self.assertRaisesRegex(ValueError, message):
                    self.repository.load(self.target)

    def test_read_rejects_rgb_or_unlisted_images(self):
        self.repository.save(self.target, {"view_00.png": (self.camera, self.rgba)})
        path = self.target / "view_00.png"
        with Image.open(path) as rgba:
            rgb = rgba.convert("RGB")
        rgb.save(path)
        with self.assertRaisesRegex(ValueError, "RGBA PNG"):
            self.repository.load(self.target)
        rgb.save(self.target / "unexpected.png")
        with self.assertRaisesRegex(ValueError, "must match"):
            self.repository.load(self.target)


class SubprocessTests(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.folder = Path(folder.name)
        self.runner = self.folder / "runner.py"
        self.runner.touch()
        self.photo = self.folder / "source photo.jpg"
        self.backend = generator.Zero123SubprocessBackend(sys.executable, self.runner, steps=3, vae_tiling=True)
        self.report = dict(status="succeeded", succeeded=1, model=generator.MODEL_ID,
                           model_revision=generator.MODEL_REVISION, seed=42,
                           pipeline=generator.PIPELINE_ID, pipeline_revision=generator.PIPELINE_REVISION,
                           photos=[dict(status="succeeded", grid="01-source photo-grid.png")])

    def fake_run(self, command, **kwargs):
        self.assertEqual(command[:3], [str(Path(sys.executable).resolve()), "-u", str(self.runner)])
        self.assertEqual(command[command.index("--photo") + 1], str(self.photo))
        self.assertEqual(command[command.index("--seed") + 1], "42")
        self.assertIn("--vae-tiling", command)
        output = Path(command[command.index("--output-dir") + 1])
        self.temporary = output.parent
        output.mkdir()
        (output / "run.json").write_text(json.dumps(self.report), encoding="utf-8")
        Image.fromarray(make_grid()).save(output / "01-source photo-grid.png")

    def test_backend_returns_grid_and_cleans_temporary_directory(self):
        with patch.object(generator.subprocess, "run", side_effect=self.fake_run):
            np.testing.assert_array_equal(self.backend(self.photo, 42), make_grid())
        self.assertFalse(self.temporary.exists())

    def test_backend_rejects_wrong_model_pipeline_or_seed(self):
        valid = self.report.copy()
        for field, value in (("model", "sudo-ai/zero123plus-v1.1"), ("model_revision", "bad"),
                             ("pipeline", "other-pipeline"), ("pipeline_revision", "bad"), ("seed", 43)):
            self.report = dict(valid, **{field: value})
            with self.subTest(field=field), patch.object(generator.subprocess, "run", side_effect=self.fake_run):
                with self.assertRaisesRegex(RuntimeError, "expected successful v1.2 result"):
                    self.backend(self.photo, 42)
                self.assertFalse(self.temporary.exists())

    def test_backend_rejects_missing_pipeline_metadata(self):
        valid = self.report.copy()
        for field in ("pipeline", "pipeline_revision"):
            self.report = valid.copy()
            del self.report[field]
            with self.subTest(field=field), patch.object(generator.subprocess, "run", side_effect=self.fake_run):
                with self.assertRaisesRegex(RuntimeError, "Invalid Zero123\\+\\+ runner output"):
                    self.backend(self.photo, 42)
            self.assertFalse(self.temporary.exists())

    def test_backend_propagates_runner_diagnostic(self):
        error = subprocess.CalledProcessError(1, "runner.py", stderr="CUDA out of memory")
        with patch.object(generator.subprocess, "run", side_effect=error):
            with self.assertRaisesRegex(RuntimeError, "CUDA out of memory") as raised:
                self.backend(self.photo, 42)
        self.assertIs(raised.exception.__cause__, error)


if __name__ == "__main__":
    unittest.main()
