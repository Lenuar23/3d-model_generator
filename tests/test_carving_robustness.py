import importlib
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
models = importlib.import_module("3d_model_soft.domain.models")
carving = importlib.import_module("3d_model_soft.services.carving")
synthetic = importlib.import_module("3d_model_soft.services.synthetic")


class RobustCarvingTests(unittest.TestCase):
    def test_known_vote_counts_for_every_threshold(self):
        camera = models.Camera(0, 0, 4, 90, 9, 9)
        # Three centers at x=-1,0,1 project to columns 3,4,5 on row 4.
        grid = carving.create_grid((-1.5, -0.5, -0.5), (1.5, 0.5, 0.5), (3, 1, 1))
        masks = []
        for columns in ((3, 4, 5), (4, 5), (5,)):
            mask = np.zeros((9, 9), dtype=bool)
            mask[4, list(columns)] = True
            masks.append(models.View(camera, mask))
        for k, expected in ((1, [True, True, True]), (2, [False, True, True]),
                            (3, [False, False, True])):
            with self.subTest(k=k):
                result = carving.carve(masks, grid, min_views=k, chunk_size=2)
                np.testing.assert_array_equal(result.occupancy[:, 0, 0], expected)

    def test_one_failed_mask_can_be_tolerated(self):
        grid = carving.create_grid(resolution=24)
        views = [models.View(cam, synthetic.ellipsoid_mask(cam))
                 for cam in synthetic.zero123_cameras(size=128)]
        failed = models.View(views[0].camera, np.zeros_like(views[0].mask))
        damaged = [failed, *views[1:]]
        self.assertFalse(carving.carve(damaged, grid).occupancy.any())
        robust = carving.carve(damaged, grid, min_views=5)
        # With one all-false view, 5-of-6 is exactly the intersection of the other 5.
        expected = carving.carve(views[1:], grid)
        np.testing.assert_array_equal(robust.occupancy, expected.occupancy)
        self.assertTrue(robust.occupancy.any())
        self.assertTrue(grid.occupancy.all())
        self.assertFalse(failed.mask.any())

    def test_defaults_remain_strict_and_thresholds_are_nested(self):
        grid = carving.create_grid(resolution=20)
        views = [models.View(cam, synthetic.ellipsoid_mask(cam))
                 for cam in synthetic.zero123_cameras(size=65)]
        baseline = carving.carve(views, grid).occupancy
        np.testing.assert_array_equal(baseline, carving.carve(views, grid, min_views=6).occupancy)
        previous = np.ones_like(baseline)
        for k in range(1, 7):
            actual = carving.carve(views, grid, min_views=k).occupancy
            self.assertFalse((actual & ~previous).any())
            previous = actual

    def test_view_order_and_chunks_do_not_change_votes(self):
        grid = carving.create_grid(resolution=17)
        views = [models.View(cam, synthetic.ellipsoid_mask(cam))
                 for cam in synthetic.zero123_cameras(size=65)]
        for k in (1, 3, 5, 6):
            baseline = carving.carve(views, grid, min_views=k)
            shuffled = carving.carve([views[i] for i in (5, 1, 3, 0, 4, 2)], grid,
                                     min_views=k, chunk_size=31)
            np.testing.assert_array_equal(baseline.occupancy, shuffled.occupancy)

    def test_bad_projection_is_zero_vote_not_a_veto(self):
        camera = models.Camera(0, 0, 4, 90, 9, 9)
        outside = models.Camera(0, 0, 4, 1, 9, 9)
        behind = models.Camera(0, 0, 0.1, 90, 9, 9)
        grid = models.VoxelGrid(np.ones((1, 1, 1), dtype=bool), (0.9, -0.1, 0.4), (0.2, 0.2, 0.2))
        views = [models.View(cam, np.ones((9, 9), dtype=bool)) for cam in (camera, outside, behind)]
        self.assertTrue(carving.carve(views, grid, min_views=1).occupancy.item())
        self.assertFalse(carving.carve(views, grid, min_views=2).occupancy.item())

    def test_removed_cells_are_not_revived(self):
        grid = carving.create_grid(resolution=2)
        grid.occupancy[0] = False
        camera = models.Camera(0, 0, 4, 90, 9, 9)
        view = models.View(camera, np.ones((9, 9), dtype=bool))
        result = carving.carve([view, view], grid, min_views=1)
        np.testing.assert_array_equal(result.occupancy, grid.occupancy)

    def test_vote_counter_handles_more_than_255_views(self):
        grid = carving.create_grid(resolution=1)
        camera = models.Camera(0, 0, 4, 90, 9, 9)
        good = models.View(camera, np.ones((9, 9), dtype=bool))
        bad = models.View(camera, np.zeros((9, 9), dtype=bool))
        views = [good] * 299 + [bad]
        self.assertTrue(carving.carve(views, grid, min_views=299).occupancy.item())
        self.assertFalse(carving.carve(views, grid, min_views=300).occupancy.item())

    def test_partial_mask_failure_recovers_shape_with_five_votes(self):
        grid = carving.create_grid(resolution=32)
        views = [models.View(cam, synthetic.ellipsoid_mask(cam))
                 for cam in synthetic.zero123_cameras(size=128)]
        originals = [view.mask.copy() for view in views]
        damaged_mask = views[0].mask.copy()
        damaged_mask[:, :64] = False
        damaged = [models.View(views[0].camera, damaged_mask), *views[1:]]
        indices = np.moveaxis(np.indices(grid.occupancy.shape), 0, -1)
        points = np.asarray(grid.origin) + (indices + 0.5) * grid.spacing
        truth = synthetic.ellipsoid_occupancy(points)
        strict = carving.carve(damaged, grid).occupancy
        robust = carving.carve(damaged, grid, min_views=5).occupancy
        strict_iou = (strict & truth).sum() / (strict | truth).sum()
        robust_iou = (robust & truth).sum() / (robust | truth).sum()
        self.assertGreater(robust_iou - strict_iou, 0.25)
        self.assertGreater((robust & truth).sum() / truth.sum(), 0.97)
        for view, original in zip(views, originals):
            np.testing.assert_array_equal(view.mask, original)

    def test_invalid_thresholds(self):
        grid = carving.create_grid(resolution=1)
        camera = models.Camera(0, 0, 4, 90, 9, 9)
        view = models.View(camera, np.ones((9, 9), dtype=bool))
        for k in (0, -1, 2, True, 1.0, np.nan, "1"):
            with self.subTest(k=k), self.assertRaises(ValueError):
                carving.carve([view], grid, min_views=k)


if __name__ == "__main__":
    unittest.main()
