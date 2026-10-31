import importlib
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
models = importlib.import_module("3d_model_soft.domain.models")
meshing = importlib.import_module("3d_model_soft.services.meshing")
carving = importlib.import_module("3d_model_soft.services.carving")
synthetic = importlib.import_module("3d_model_soft.services.synthetic")


def volume(mesh):
    a, b, c = np.moveaxis(mesh.vertices[mesh.faces], 1, 0)
    return np.einsum("ij,ij->", a, np.cross(b, c)) / 6


def ellipsoid_grid(resolution=40, radii=(0.6, 0.8, 0.5)):
    grid = carving.create_grid(resolution=resolution)
    points = np.moveaxis(np.indices(grid.occupancy.shape), 0, -1)
    points = np.asarray(grid.origin) + (points + 0.5) * grid.spacing
    return models.VoxelGrid(synthetic.ellipsoid_occupancy(points, radii), grid.origin, grid.spacing)


class MeshingTests(unittest.TestCase):
    def assert_closed_oriented(self, mesh):
        edges = np.concatenate((mesh.faces[:, [0, 1]], mesh.faces[:, [1, 2]], mesh.faces[:, [2, 0]]))
        _, inverse, counts = np.unique(np.sort(edges, axis=1), axis=0, return_inverse=True, return_counts=True)
        self.assertTrue((counts == 2).all())
        winding = np.bincount(inverse, weights=np.where(edges[:, 0] < edges[:, 1], 1, -1))
        self.assertTrue((winding == 0).all())

    def test_empty_grid(self):
        grid = models.VoxelGrid(np.zeros((3, 3, 3), dtype=bool), (0, 0, 0), (1, 1, 1))
        mesh = meshing.build_mesh(grid)
        self.assertEqual(mesh.vertices.shape, (0, 3))
        self.assertEqual(mesh.faces.shape, (0, 3))
        self.assertEqual(meshing.vertex_normals(mesh).shape, (0, 3))

    def test_one_voxel_is_closed_with_world_coordinates(self):
        grid = models.VoxelGrid(np.ones((1, 1, 1), dtype=bool), (10, -3, 5), (2, 4, 6))
        mesh = meshing.marching_cubes(grid)
        self.assertEqual(len(mesh.vertices), 6)
        self.assertEqual(len(mesh.faces), 8)
        np.testing.assert_allclose(mesh.vertices.min(axis=0), grid.origin)
        np.testing.assert_allclose(mesh.vertices.max(axis=0), (12, 1, 11))
        self.assert_closed_oriented(mesh)
        self.assertGreater(volume(mesh), 0)

    def test_full_grid_is_closed_at_boundary(self):
        grid = carving.create_grid((2, -1, 10), (4, 3, 16), (4, 5, 6))
        mesh = meshing.marching_cubes(grid)
        np.testing.assert_allclose(mesh.vertices.min(axis=0), (2, -1, 10))
        np.testing.assert_allclose(mesh.vertices.max(axis=0), (4, 3, 16))
        self.assert_closed_oriented(mesh)
        self.assertGreater(volume(mesh), 0)

    def test_all_256_cases_have_valid_triangles(self):
        for case in range(256):
            occupancy = np.zeros((2, 2, 2), dtype=bool)
            for bit, corner in enumerate(meshing._CORNERS):
                occupancy[tuple(corner)] = bool(case & (1 << bit))
            mesh = meshing.marching_cubes(models.VoxelGrid(occupancy, (0, 0, 0), (1, 1, 1)))
            triangles = mesh.vertices[mesh.faces]
            areas = np.linalg.norm(np.cross(triangles[:, 1] - triangles[:, 0],
                                            triangles[:, 2] - triangles[:, 0]), axis=1)
            with self.subTest(case=case):
                self.assertTrue((areas > 0).all())
                self.assertEqual(len(np.unique(mesh.vertices, axis=0)), len(mesh.vertices))

    def test_ellipsoid_volume_normals_and_topology(self):
        grid = ellipsoid_grid()
        mesh = meshing.marching_cubes(grid)
        expected = 4 / 3 * np.pi * 0.6 * 0.8 * 0.5
        self.assertAlmostEqual(volume(mesh) / expected, 1, delta=0.02)
        self.assert_closed_oriented(mesh)
        edges = meshing._edges(mesh.faces)
        self.assertEqual(len(mesh.vertices) - len(edges) + len(mesh.faces), 2)
        normals = meshing.vertex_normals(mesh)
        np.testing.assert_allclose(np.linalg.norm(normals, axis=1), 1)
        self.assertTrue((np.einsum("ij,ij->i", mesh.vertices, normals) > 0).all())

    def test_small_components_removed_without_losing_main_object(self):
        grid = carving.create_grid(resolution=16)
        grid.occupancy[:] = False
        grid.occupancy[4:12, 4:12, 4:12] = True
        grid.occupancy[1, 1, 1] = True
        raw = meshing.marching_cubes(grid)
        clean = meshing.clean_mesh(raw, min_component_faces=9, min_component_ratio=0)
        self.assertEqual(len(raw.faces) - len(clean.faces), 8)
        self.assert_closed_oriented(clean)
        self.assertEqual(len(raw.faces), len(meshing.clean_mesh(raw, min_component_ratio=0).faces))
        self.assertEqual(len(meshing.clean_mesh(raw, min_component_faces=100000).faces), 0)

    def test_cleanup_duplicates_and_degenerates(self):
        vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [3, 3, 3]], dtype=float)
        mesh = models.Mesh(vertices, np.array([[0, 1, 2], [2, 1, 0], [0, 0, 1]]))
        clean = meshing.clean_mesh(mesh)
        self.assertEqual(clean.faces.shape, (1, 3))
        self.assertEqual(clean.vertices.shape, (3, 3))
        self.assertEqual(mesh.faces.shape, (3, 3))

    def test_smoothing_preserves_topology_and_limits_volume_change(self):
        mesh = meshing.marching_cubes(ellipsoid_grid())
        original = mesh.vertices.copy()
        smoothed = meshing.smooth_mesh(mesh)
        np.testing.assert_array_equal(smoothed.faces, mesh.faces)
        np.testing.assert_array_equal(mesh.vertices, original)
        self.assertFalse(np.array_equal(smoothed.vertices, original))
        self.assertAlmostEqual(volume(smoothed) / volume(mesh), 1, delta=0.02)
        edges = meshing._edges(mesh.faces)
        self.assertLess(np.linalg.norm(smoothed.vertices[edges[:, 0]] - smoothed.vertices[edges[:, 1]], axis=1).mean(),
                        np.linalg.norm(mesh.vertices[edges[:, 0]] - mesh.vertices[edges[:, 1]], axis=1).mean())

    def test_open_boundary_and_isolated_vertices_are_fixed(self):
        mesh = models.Mesh(np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [9, 9, 9]], dtype=float),
                           np.array([[0, 1, 2]]))
        np.testing.assert_array_equal(meshing.smooth_mesh(mesh).vertices, mesh.vertices)
        np.testing.assert_array_equal(meshing.vertex_normals(mesh)[3], [0, 0, 0])

    def test_carving_to_mesh(self):
        cameras = synthetic.zero123_cameras(size=128)
        views = [models.View(cam, synthetic.ellipsoid_mask(cam)) for cam in cameras]
        grid = carving.carve(views, carving.create_grid(resolution=32))
        mesh = meshing.build_mesh(grid)
        self.assertGreater(len(mesh.faces), 100)
        self.assert_closed_oriented(mesh)
        expected = 4 / 3 * np.pi * 0.6 * 0.8 * 0.5
        self.assertAlmostEqual(volume(mesh) / expected, 1, delta=0.15)

    def test_invalid_parameters(self):
        mesh = meshing.marching_cubes(ellipsoid_grid(resolution=8))
        for options in ({"iterations": -1}, {"iterations": True}, {"lambda_factor": 1},
                        {"mu_factor": -0.1}, {"mu_factor": np.nan}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                meshing.smooth_mesh(mesh, **options)
        for options in ({"min_component_faces": -1}, {"min_component_ratio": 1.1}):
            with self.assertRaises(ValueError):
                meshing.clean_mesh(mesh, **options)
        with self.assertRaises(ValueError):
            models.Mesh(np.zeros((1, 3)), np.array([[0, 1, 2]]))


if __name__ == "__main__":
    unittest.main()
