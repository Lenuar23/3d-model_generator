import importlib
import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest
import zlib

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
synthetic = importlib.import_module("3d_model_soft.services.synthetic")
dataset = importlib.import_module("3d_model_soft.infrastructure.synthetic_dataset")
project = importlib.import_module("3d_model_soft.services.camera_math").project
Camera = importlib.import_module("3d_model_soft.domain.models").Camera


def read_png(path):
    """Decode the generated subset of PNG and independently check chunk CRCs."""
    data = path.read_bytes()
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    offset, compressed = 8, b""
    while offset < len(data):
        length = struct.unpack(">I", data[offset:offset + 4])[0]
        kind = data[offset + 4:offset + 8]
        payload = data[offset + 8:offset + 8 + length]
        crc = struct.unpack(">I", data[offset + 8 + length:offset + 12 + length])[0]
        assert crc == zlib.crc32(kind + payload)
        if kind == b"IHDR":
            width, height, bits, color, compression, filtering, interlace = struct.unpack(">IIBBBBB", payload)
            assert (bits, color, compression, filtering, interlace) == (8, 6, 0, 0, 0)
        elif kind == b"IDAT":
            compressed += payload
        offset += length + 12
    rows = np.frombuffer(zlib.decompress(compressed), dtype=np.uint8).reshape(height, width * 4 + 1)
    assert (rows[:, 0] == 0).all()
    return rows[:, 1:].reshape(height, width, 4)


class SyntheticTests(unittest.TestCase):
    def test_published_angles(self):
        for version, elevations in (("v1.1", [30, -20] * 3), ("v1.2", [20, -10] * 3)):
            cameras = synthetic.zero123_cameras(version)
            self.assertEqual([cam.azimuth_deg for cam in cameras], [30, 90, 150, 210, 270, 330])
            self.assertEqual([cam.elevation_deg for cam in cameras], elevations)

    def test_sphere_silhouette_matches_analytic_disk(self):
        # Tangent cone: projected radius = focal * radius / sqrt(distance^2-radius^2).
        cam = Camera(0, 0, 4, 40, 161, 101)
        mask = synthetic.ellipsoid_mask(cam, (0.7, 0.7, 0.7))
        focal = 101 / (2 * np.tan(np.radians(20)))
        radius_squared = (focal * 0.7)**2 / (4**2 - 0.7**2)
        v, u = np.indices(mask.shape)
        np.testing.assert_array_equal(mask, (u - 80)**2 + (v - 50)**2 <= radius_squared)

    def test_inside_points_project_into_all_masks(self):
        rng = np.random.default_rng(42)
        points = rng.uniform(-0.15, 0.15, (100, 3))
        for cam in synthetic.zero123_cameras(size=129):
            mask = synthetic.ellipsoid_mask(cam)
            pixels = np.rint(project(points, cam)).astype(int)
            self.assertTrue(mask[pixels[:, 1], pixels[:, 0]].all())
            self.assertTrue(mask.any())
            self.assertFalse(mask[[0, -1], :].any())
            self.assertFalse(mask[:, [0, -1]].any())

    def test_ellipsoid_changes_with_view(self):
        cameras = synthetic.zero123_cameras(size=129)
        self.assertFalse(np.array_equal(synthetic.ellipsoid_mask(cameras[0]),
                                       synthetic.ellipsoid_mask(cameras[1])))

    def test_occupancy_boundary_and_volume(self):
        np.testing.assert_array_equal(
            synthetic.ellipsoid_occupancy([[0, 0, 0], [0.6, 0, 0], [0.61, 0, 0]]),
            [True, True, False])
        axis = (np.arange(100) + 0.5) * 0.02 - 1
        points = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1)
        volume = synthetic.ellipsoid_occupancy(points).sum() * 0.02**3
        self.assertAlmostEqual(volume / (4 / 3 * np.pi * 0.6 * 0.8 * 0.5), 1, delta=0.01)

    def test_dataset_round_trip_and_repeatability(self):
        with tempfile.TemporaryDirectory() as temp:
            first = dataset.generate_dataset(Path(temp) / "first", size=65, resolution=24)
            second = dataset.generate_dataset(Path(temp) / "second", size=65, resolution=24)
            metadata = json.loads((first / "views/cameras.json").read_text())
            self.assertEqual(len(metadata), 6)
            for (name, values), cam in zip(metadata.items(), synthetic.zero123_cameras(size=65)):
                self.assertEqual(set(values), {"azimuth", "elevation", "distance", "fov"})
                image = read_png(first / "views" / name)
                np.testing.assert_array_equal(image[..., 3] == 255, synthetic.ellipsoid_mask(cam))
                self.assertEqual((first / "views" / name).read_bytes(),
                                 (second / "views" / name).read_bytes())
            with np.load(first / "ground_truth.npz") as truth:
                self.assertEqual(truth["occupancy"].shape, (24, 24, 24))
                self.assertEqual(truth["occupancy"].dtype, np.bool_)
                points = np.stack(np.meshgrid(truth["x"], truth["y"], truth["z"], indexing="ij"), axis=-1)
                np.testing.assert_array_equal(truth["occupancy"], synthetic.ellipsoid_occupancy(points))
            with self.assertRaises(FileExistsError):
                dataset.generate_dataset(first)

    def test_invalid_inputs_leave_no_dataset(self):
        for radii in ((0, 1, 1), (-1, 1, 1), (1, np.nan, 1), (1, 2)):
            with self.subTest(radii=radii), self.assertRaises(ValueError):
                synthetic.ellipsoid_mask(Camera(0, 0, 4, 30, 33, 33), radii)
        with self.assertRaises(ValueError):
            synthetic.zero123_cameras("unknown")
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "invalid"
            for options in ({"size": 0}, {"resolution": 1}, {"radii": (0, 1, 1)}):
                with self.subTest(options=options), self.assertRaises(ValueError):
                    dataset.generate_dataset(output, **options)
                self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
