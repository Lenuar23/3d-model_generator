import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np
from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from texture_bake import triangle_pixels, bake_texture

class RasterTests(unittest.TestCase):
    def test_barycentric_coordinates_and_bounds(self):
        p = np.array([[-2., 0], [4, 0], [0, 4]])
        y, x, weights = triangle_pixels(p, 3, 3)
        np.testing.assert_allclose(weights.sum(axis=1), 1)
        np.testing.assert_allclose(weights @ p, np.column_stack([x, y]), atol=1e-7)
        self.assertTrue(((x >= 0) & (x < 3) & (y >= 0) & (y < 3)).all())
        self.assertIsNone(triangle_pixels(np.zeros((3, 2)), 3, 3))

@unittest.skipUnless(importlib.util.find_spec("xatlas"), "Needs texture runtime")
class BakeTests(unittest.TestCase):
    def test_roundtrip_has_texture_and_preserves_surface(self):
        import trimesh
        mesh = trimesh.creation.icosphere(subdivisions=1)
        original = mesh.vertices.copy()
        photo = Image.new("RGBA", (32, 32), (180, 30, 10, 255))
        output, atlas = bake_texture(mesh, photo, Image.new("RGB", (16, 16), (180, 30, 10)), size=64)
        np.testing.assert_array_equal(mesh.vertices, original)
        self.assertEqual(len(output.faces), len(mesh.faces))
        np.testing.assert_allclose(output.bounds, mesh.bounds)
        self.assertTrue(np.isfinite(output.visual.uv).all())
        self.assertLessEqual(np.abs(np.asarray(atlas).astype(int)-[180,30,10]).max(), 1)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/"model.glb"
            output.export(path)
            restored = trimesh.load(path, force="mesh", process=False)
            self.assertEqual(restored.visual.kind, "texture")
            self.assertIsNotNone(restored.visual.material.baseColorTexture)

    def test_empty_foreground_rejected(self):
        import trimesh
        with self.assertRaisesRegex(ValueError, "foreground"):
            bake_texture(trimesh.creation.box(), Image.new("RGBA", (8,8)), Image.new("RGB", (8,8)), size=32)
