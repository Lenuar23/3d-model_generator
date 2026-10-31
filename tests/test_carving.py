import importlib
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
models = importlib.import_module("3d_model_soft.domain.models")
carving = importlib.import_module("3d_model_soft.services.carving")
synthetic = importlib.import_module("3d_model_soft.services.synthetic")


def synthetic_result(radii=(0.6, 0.8, 0.5), version="v1.2", chunk_size=65536):
    grid = carving.create_grid(resolution=48)
    cameras = synthetic.zero123_cameras(version, size=256)
    views = [models.View(cam, synthetic.ellipsoid_mask(cam, radii)) for cam in cameras]
    result = carving.carve(views, grid, chunk_size=chunk_size)
    indices = np.moveaxis(np.indices(grid.occupancy.shape), 0, -1)
    points = np.asarray(grid.origin) + (indices + 0.5) * grid.spacing
    truth = synthetic.ellipsoid_occupancy(points, radii)
    return result.occupancy, truth


class CarvingTests(unittest.TestCase):
    def test_grid_uses_cell_centers_and_xyz_axes(self):
        grid = carving.create_grid((-1, -2, -3), (1, 2, 3), (2, 4, 3))
        self.assertEqual(grid.occupancy.shape, (2, 4, 3))
        np.testing.assert_allclose(grid.spacing, (1, 1, 2))
        np.testing.assert_allclose(np.asarray(grid.origin) + np.asarray(grid.spacing) / 2,
                                   (-0.5, -1.5, -2))
        self.assertTrue(grid.occupancy.all())

    def test_empty_and_full_masks_and_no_mutation(self):
        cam = models.Camera(0, 0, 4, 90, 9, 9)
        grid = carving.create_grid(resolution=5)
        for foreground in (False, True):
            mask = np.full((9, 9), foreground)
            result = carving.carve([models.View(cam, mask)], grid)
            self.assertTrue((result.occupancy == foreground).all())
            self.assertTrue((mask == foreground).all())
            self.assertTrue(grid.occupancy.all())

    def test_mask_uv_order(self):
        cam = models.Camera(0, 0, 4, 90, 9, 9)
        mask = np.zeros((9, 9), dtype=bool)
        mask[3, 5] = True
        # Cell centers have x,y = +/-1, z=0: only (+1,+1,0) hits (u=5,v=3).
        grid = carving.create_grid((-2, -2, -0.1), (2, 2, 0.1), (2, 2, 1))
        result = carving.carve([models.View(cam, mask)], grid)
        expected = np.zeros((2, 2, 1), dtype=bool)
        expected[1, 1, 0] = True
        np.testing.assert_array_equal(result.occupancy, expected)

    def test_invalid_depth_and_outside_frame_are_carved(self):
        cam = models.Camera(0, 0, 4, 90, 9, 9)
        view = models.View(cam, np.ones((9, 9), dtype=bool))
        for center in ((0, 0, 4), (0, 0, 5), (100, 0, 0), (0, 100, 0)):
            point = np.asarray(center)
            grid = carving.create_grid(tuple(point - 0.01), tuple(point + 0.01), 1)
            self.assertFalse(carving.carve([view], grid).occupancy.any())

    def test_pixel_footprint_at_image_edge(self):
        cam = models.Camera(0, 0, 4, 90, 8, 8)
        view = models.View(cam, np.ones((8, 8), dtype=bool))
        # For z=0 focal/depth=1, so u=x+3.5; left edge is x=-4.
        for x, expected in ((-3.9, True), (-4.1, False), (3.9, True), (4.1, False)):
            grid = models.VoxelGrid(np.ones((1, 1, 1), dtype=bool),
                                    (x - 0.05, -0.05, -0.05), (0.1, 0.1, 0.1))
            self.assertEqual(bool(carving.carve([view], grid).occupancy.item()), expected)

    def test_existing_empty_cells_stay_empty_and_views_intersect(self):
        cam = models.Camera(0, 0, 4, 90, 9, 9)
        grid = carving.create_grid(resolution=4)
        grid.occupancy[0] = False
        view = models.View(cam, np.ones((9, 9), dtype=bool))
        result = carving.carve([view], grid, chunk_size=7)
        np.testing.assert_array_equal(result.occupancy, grid.occupancy)
        empty = models.View(cam, np.zeros((9, 9), dtype=bool))
        self.assertFalse(carving.carve([view, empty], grid).occupancy.any())

    def test_chunk_size_does_not_change_result(self):
        grid = carving.create_grid(resolution=17)
        views = [models.View(cam, synthetic.ellipsoid_mask(cam))
                 for cam in synthetic.zero123_cameras(size=65)]
        baseline = carving.carve(views, grid, chunk_size=10000)
        chunked = carving.carve(views, grid, chunk_size=31)
        np.testing.assert_array_equal(baseline.occupancy, chunked.occupancy)

    def test_synthetic_reconstruction_quality(self):
        for version in ("v1.1", "v1.2"):
            for radii in ((0.65, 0.65, 0.65), (0.6, 0.8, 0.5)):
                with self.subTest(version=version, radii=radii):
                    actual, truth = synthetic_result(radii, version)
                    intersection = (actual & truth).sum()
                    iou = intersection / (actual | truth).sum()
                    recall = intersection / truth.sum()
                    self.assertGreater(iou, 0.85)
                    self.assertGreater(recall, 0.96)
                    self.assertFalse(actual[[0, -1]].any())

    def test_invalid_inputs(self):
        cam = models.Camera(0, 0, 4, 90, 9, 9)
        grid = carving.create_grid(resolution=2)
        view = models.View(cam, np.ones((9, 9), dtype=bool))
        for options in ({"resolution": 0}, {"resolution": True}, {"resolution": 2.5}, {"resolution": (1, 2)},
                        {"resolution": (1, 2.5, 3)}, {"bounds_min": (0, 0, 0), "bounds_max": (0, 1, 1)},
                        {"bounds_min": (np.nan, 0, 0)}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                carving.create_grid(**options)
        for mask in (np.ones((9, 9), dtype=np.uint8), np.ones((8, 9), dtype=bool)):
            with self.assertRaises(ValueError):
                models.View(cam, mask)
        for chunk_size in (0, -1, True, 2.5):
            with self.assertRaises(ValueError):
                carving.carve([view], grid, chunk_size=chunk_size)
        with self.assertRaises(ValueError):
            carving.carve([], grid)
        with self.assertRaises(ValueError):
            models.VoxelGrid(np.ones((2, 2), dtype=bool), (0, 0, 0), (1, 1, 1))
        with self.assertRaises(ValueError):
            models.VoxelGrid(np.ones((2, 2, 2), dtype=bool), (0, 0, 0), (0, 1, 1))


if __name__ == "__main__":
    unittest.main()
