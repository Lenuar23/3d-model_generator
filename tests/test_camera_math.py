"""Analytic projection checks, runnable with the standard-library unittest."""

import importlib
from pathlib import Path
import sys
import unittest

import numpy as np

# The existing package name starts with a digit; importlib supports that name.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
Camera = importlib.import_module("3d_model_soft.domain.models").Camera
camera_math = importlib.import_module("3d_model_soft.services.camera_math")


def camera(**overrides):
    values = dict(azimuth_deg=0, elevation_deg=0, distance=4,
                  fov_deg=90, width=201, height=101)
    return Camera(**(values | overrides))


class CameraMathTests(unittest.TestCase):
    def test_origin_is_center_for_all_orientations(self):
        for azimuth in (0, 90, 180, 270, 390):
            for elevation in (-90, -30, 0, 20, 90):
                with self.subTest(azimuth=azimuth, elevation=elevation):
                    cam = camera(azimuth_deg=azimuth, elevation_deg=elevation)
                    np.testing.assert_allclose(
                        camera_math.project([[0, 0, 0]], cam), [[100, 50]])
                    rotation, translation = camera_math.extrinsics(cam)
                    np.testing.assert_allclose(rotation @ rotation.T, np.eye(3), atol=1e-14)
                    self.assertAlmostEqual(np.linalg.det(rotation), 1)
                    np.testing.assert_allclose(translation, [0, 0, 4], atol=1e-14)

    def test_axes_and_vertical_fov_on_rectangular_image(self):
        points = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 2], [1, 0, 2]]
        np.testing.assert_allclose(camera_math.project(points, camera()),
                                   [[100, 50], [112.625, 50], [100, 37.375],
                                    [100, 50], [125.25, 50]])

    def test_quarter_turn_camera(self):
        np.testing.assert_allclose(
            camera_math.project([[0, 0, 1], [0, 1, 0]], camera(azimuth_deg=90)),
            [[87.375, 50], [100, 37.375]])

    def test_elevation_orientation(self):
        np.testing.assert_allclose(
            camera_math.project([[0, 0, 1]], camera(elevation_deg=30)),
            [[100, 50 + 25.25 / (4 - np.sqrt(3) / 2)]])

    def test_invalid_depth_is_nan(self):
        pixels = camera_math.project([[0, 0, 4], [1, 1, 5], [0, 0, 0]], camera())
        self.assertTrue(np.isnan(pixels[:2]).all())
        self.assertTrue(np.isfinite(pixels[2]).all())

    def test_outside_image_is_not_clipped(self):
        pixels = camera_math.project([[100, 0, 0]], camera())
        self.assertGreater(pixels[0, 0], 201)

    def test_even_image_center(self):
        np.testing.assert_allclose(
            camera_math.project([[0, 0, 0]], camera(width=200, height=100)), [[99.5, 49.5]])

    def test_empty_array(self):
        self.assertEqual(camera_math.project(np.empty((0, 3)), camera()).shape, (0, 2))

    def test_invalid_points(self):
        for points in ([], [1, 2, 3], [[1, 2]], [[[1, 2, 3]]],
                       [[0, np.nan, 0]], [[np.inf, 0, 0]]):
            with self.subTest(points=points), self.assertRaises(ValueError):
                camera_math.project(points, camera())

    def test_invalid_camera_parameters(self):
        for overrides in ({"distance": 0}, {"distance": -1}, {"distance": np.inf},
                          {"fov_deg": 0}, {"fov_deg": 180}, {"fov_deg": np.nan},
                          {"elevation_deg": 91}, {"elevation_deg": -91},
                          {"azimuth_deg": np.inf}, {"width": 0}, {"height": -1},
                          {"width": 10.5}, {"height": True}):
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                camera(**overrides)


if __name__ == "__main__":
    unittest.main()
