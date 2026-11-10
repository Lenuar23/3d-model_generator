import importlib
import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
reconstruct_mod = importlib.import_module("3d_model_soft.app.reconstruct")
container_mod = importlib.import_module("3d_model_soft.app.container")
synthetic = importlib.import_module("3d_model_soft.services.synthetic")


def _create_synthetic_bundle(target_dir: Path, size: int = 64) -> Path:
    target_dir.mkdir(parents=True, exist_ok=True)
    cameras = synthetic.zero123_cameras(size=size)
    metadata = {}
    for index, cam in enumerate(cameras):
        mask = synthetic.ellipsoid_mask(cam)
        # Create synthetic RGBA image with colored ellipse
        rgba = np.zeros((size, size, 4), dtype=np.uint8)
        rgba[mask, 0] = 220
        rgba[mask, 1] = 60
        rgba[mask, 2] = 40
        rgba[mask, 3] = 255
        filename = f"view_{index:02d}.png"
        Image.fromarray(rgba).save(target_dir / filename)
        metadata[filename] = {
            "azimuth": float(cam.azimuth_deg),
            "elevation": float(cam.elevation_deg),
            "distance": float(cam.distance),
            "fov": float(cam.fov_deg),
        }
    (target_dir / "cameras.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return target_dir


class ReconstructScenarioTests(unittest.TestCase):
    def test_reconstruct_end_to_end_glb(self):
        with tempfile.TemporaryDirectory() as tmp:
            views_dir = _create_synthetic_bundle(Path(tmp) / "views", size=64)
            output_glb = Path(tmp) / "output" / "test_model.glb"

            result = reconstruct_mod.reconstruct(views_dir, output_glb, resolution=32, smoothing_iterations=2)
            self.assertEqual(result, output_glb.resolve())
            self.assertTrue(output_glb.exists())
            self.assertGreater(output_glb.stat().st_size, 1000)

            # Verify valid GLB header
            raw = output_glb.read_bytes()
            magic, version, total_len = struct.unpack_from("<4sII", raw)
            self.assertEqual(magic, b"glTF")
            self.assertEqual(version, 2)
            self.assertEqual(total_len, len(raw))

    def test_reconstruct_default_output_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            views_dir = _create_synthetic_bundle(Path(tmp) / "views", size=48)
            result = reconstruct_mod.reconstruct(views_dir, resolution=24)
            expected = views_dir.resolve() / "model.glb"
            self.assertEqual(result, expected)
            self.assertTrue(expected.exists())

    def test_container_dispatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            views_dir = _create_synthetic_bundle(Path(tmp) / "views", size=48)
            container = container_mod.Container.default()
            result = container.reconstruct(views_dir, resolution=24)
            self.assertTrue(result.exists())


if __name__ == "__main__":
    unittest.main()
