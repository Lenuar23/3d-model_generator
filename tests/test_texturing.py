import importlib
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
models = importlib.import_module("3d_model_soft.domain.models")
texturing = importlib.import_module("3d_model_soft.services.texturing")
meshing = importlib.import_module("3d_model_soft.services.meshing")


def triangle(z=0):
    return models.Mesh(np.array([[-1, -1, z], [1, -1, z], [0, 1, z]], dtype=float),
                       np.array([[0, 1, 2]]))


def view(color=(255, 0, 0), azimuth=0):
    camera = models.Camera(azimuth, 0, 4, 90, 33, 33)
    image = np.tile(np.array(color, dtype=np.uint8), (33, 33, 1))
    return models.View(camera, np.ones((33, 33), dtype=bool), image)


class TexturingTests(unittest.TestCase):
    def test_constant_color_and_input_immutability(self):
        mesh, source = triangle(), view((10, 90, 230))
        result = texturing.texture_mesh(mesh, [source])
        np.testing.assert_array_equal(result.colors, [[10, 90, 230]] * 3)
        np.testing.assert_array_equal(result.vertices, mesh.vertices)
        self.assertIsNone(mesh.colors)
        self.assertFalse(np.shares_memory(result.vertices, mesh.vertices))
        np.testing.assert_array_equal(source.image[0, 0], [10, 90, 230])

    def test_samples_uv_in_correct_order(self):
        source = view()
        v, u = np.indices((33, 33))
        source.image[:] = np.stack((u, v, u * 0), axis=-1)
        result = texturing.texture_mesh(triangle(), [source])
        np.testing.assert_array_equal(result.colors, [[12, 20, 0], [20, 20, 0], [16, 12, 0]])

    def test_hidden_vertices_do_not_receive_color(self):
        back = triangle()
        front = triangle(z=1)
        front.vertices[:, :2] *= 2
        mesh = models.Mesh(np.concatenate((back.vertices, front.vertices)),
                           np.array([[0, 1, 2], [3, 4, 5]]))
        result = texturing.texture_mesh(mesh, [view()])
        np.testing.assert_array_equal(result.colors[:3], [[128, 128, 128]] * 3)
        np.testing.assert_array_equal(result.colors[3:], [[255, 0, 0]] * 3)

    def test_subpixel_occluder_is_detected(self):
        # A triangle smaller than one pixel still occludes the exact vertex ray.
        back = models.Mesh(np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float), np.array([[0, 1, 2]]))
        front = np.array([[-0.01, -0.01, 1], [0.01, -0.01, 1], [0, 0.01, 1]])
        mesh = models.Mesh(np.concatenate((back.vertices, front)), np.array([[0, 1, 2], [3, 4, 5]]))
        colors = texturing.texture_mesh(mesh, [view()]).colors
        np.testing.assert_array_equal(colors[0], [128, 128, 128])
        np.testing.assert_array_equal(colors[1:3], [[255, 0, 0]] * 2)

    def test_depth_interpolation_is_perspective_correct(self):
        # Query centroid of projected triangle: reciprocal depths average.
        camera = view().camera
        vertices = np.array([[-1, -1, 2], [1, -1, 0], [0, 1, 1]], dtype=float)
        mesh = models.Mesh(vertices, np.array([[0, 1, 2]]))
        projection = importlib.import_module("3d_model_soft.services.camera_math").project
        pixel = projection(vertices, camera).mean(axis=0, keepdims=True)
        depth = texturing._sample_depths(mesh, camera, pixel)
        np.testing.assert_allclose(depth, [3 / (1 / 2 + 1 / 4 + 1 / 3)])

    def test_triangle_crossing_camera_plane_still_occludes(self):
        camera = view().camera
        mesh = models.Mesh(np.array([[-1, -1, 2], [1, -1, 2], [0, 1, 5]], dtype=float),
                           np.array([[0, 1, 2]]))
        depth = texturing._sample_depths(mesh, camera, np.array([[16, 20]], dtype=float))
        self.assertTrue(np.isfinite(depth).all())
        self.assertGreater(depth[0], 0)

    def test_backfaces_and_missing_observations_use_fallback(self):
        mesh = triangle()
        for source in (view(azimuth=180), view()):
            if source.camera.azimuth_deg == 0:
                source.mask[:] = False
            result = texturing.texture_mesh(mesh, [source], fallback_color=(1, 2, 3))
            np.testing.assert_array_equal(result.colors, [[1, 2, 3]] * 3)

    def test_blending_uses_linear_rgb(self):
        result = texturing.texture_mesh(triangle(), [view((255, 0, 0)), view((0, 0, 255))])
        np.testing.assert_array_equal(result.colors, [[188, 0, 188]] * 3)

    def test_behind_camera_and_outside_frame(self):
        for mesh in (triangle(z=5), triangle()):
            if mesh.vertices[0, 2] == 0:
                mesh.vertices[:, 0] += 100
            np.testing.assert_array_equal(texturing.texture_mesh(mesh, [view()]).colors,
                                           [[128, 128, 128]] * 3)

    def test_empty_and_unreferenced_vertices(self):
        mesh = models.Mesh(np.empty((0, 3)), np.empty((0, 3), dtype=int))
        self.assertEqual(texturing.texture_mesh(mesh, [view()]).colors.shape, (0, 3))
        isolated = models.Mesh(np.array([[0, 0, 0]], dtype=float), np.empty((0, 3), dtype=int))
        np.testing.assert_array_equal(texturing.texture_mesh(isolated, [view()]).colors, [[128, 128, 128]])

    def test_mesh_processing_preserves_existing_colors(self):
        colored = texturing.texture_mesh(triangle(), [view()])
        for result in (meshing.clean_mesh(colored), meshing.smooth_mesh(colored)):
            np.testing.assert_array_equal(result.colors, colored.colors)
            self.assertFalse(np.shares_memory(result.colors, colored.colors))

    def test_invalid_inputs(self):
        for options in ({"fallback_color": (-1, 0, 0)}, {"fallback_color": (0.5, 0, 0)},
                        {"depth_tolerance": -1}, {"depth_tolerance": np.nan}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                texturing.texture_mesh(triangle(), [view()], **options)
        for sources in ([], [models.View(view().camera, view().mask)]):
            with self.assertRaises(ValueError):
                texturing.texture_mesh(triangle(), sources)
        with self.assertRaises(ValueError):
            models.View(view().camera, view().mask, np.zeros((33, 33, 4), dtype=np.uint8))
        with self.assertRaises(ValueError):
            models.Mesh(triangle().vertices, triangle().faces, np.zeros((2, 3), dtype=np.uint8))


if __name__ == "__main__":
    unittest.main()
