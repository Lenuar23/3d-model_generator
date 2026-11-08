import importlib
import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
models = importlib.import_module("3d_model_soft.domain.models")
GlbExporter = importlib.import_module("3d_model_soft.infrastructure.export.glb_exporter").GlbExporter


def read_glb(path):
    """Independent parser for the container and accessor bytes used by tests."""
    raw = path.read_bytes()
    magic, version, length = struct.unpack_from("<4sII", raw)
    assert (magic, version, length) == (b"glTF", 2, len(raw))
    json_length, json_type = struct.unpack_from("<I4s", raw, 12)
    assert json_type == b"JSON" and json_length % 4 == 0
    document = json.loads(raw[20:20 + json_length])
    bin_length, bin_type = struct.unpack_from("<I4s", raw, 20 + json_length)
    assert bin_type == b"BIN\x00" and bin_length % 4 == 0
    binary = raw[28 + json_length:]
    assert len(binary) == bin_length
    assert document["buffers"][0]["byteLength"] <= bin_length
    assert bin_length - document["buffers"][0]["byteLength"] < 4

    def accessor(index):
        spec = document["accessors"][index]
        view = document["bufferViews"][spec["bufferView"]]
        offset = view.get("byteOffset", 0) + spec.get("byteOffset", 0)
        assert offset % 4 == 0
        components = {"SCALAR": 1, "VEC3": 3}[spec["type"]]
        dtype = {5126: "<f4", 5123: "<u2", 5125: "<u4"}[spec["componentType"]]
        count = spec["count"] * components
        assert count * np.dtype(dtype).itemsize == view["byteLength"]
        assert offset + view["byteLength"] <= document["buffers"][0]["byteLength"]
        values = np.frombuffer(binary, dtype=dtype, count=count, offset=offset)
        return values.reshape(-1, components) if components > 1 else values

    return document, accessor


def triangle(colors=None):
    return models.Mesh(np.array([[1, 2, 3], [2, 2, 3], [1, 3, 3]], dtype=float),
                       np.array([[0, 1, 2]]), colors)


class GlbExporterTests(unittest.TestCase):
    def test_geometry_normals_scene_and_material_round_trip(self):
        mesh = triangle()
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "nested/model.glb"
            self.assertEqual(GlbExporter().export(mesh, path), path)
            document, read = read_glb(path)
            primitive = document["meshes"][0]["primitives"][0]
            attributes = primitive["attributes"]
            np.testing.assert_array_equal(read(attributes["POSITION"]), mesh.vertices)
            np.testing.assert_array_equal(read(attributes["NORMAL"]), [[0, 0, 1]] * 3)
            np.testing.assert_array_equal(read(primitive["indices"]), [0, 1, 2])
            self.assertNotIn("COLOR_0", attributes)
            self.assertEqual(document["asset"]["version"], "2.0")
            self.assertEqual(document["scenes"][document["scene"]]["nodes"], [0])
            self.assertEqual(document["nodes"][0]["mesh"], 0)
            self.assertEqual(primitive["mode"], 4)
            spec = document["accessors"][attributes["POSITION"]]
            self.assertEqual(spec["min"], [1, 2, 3])
            self.assertEqual(spec["max"], [2, 3, 3])
            self.assertEqual(document["materials"][0]["pbrMetallicRoughness"]["metallicFactor"], 0)

    def test_srgb_colors_are_encoded_as_linear_floats(self):
        colors = np.array([[0, 128, 255], [10, 64, 200], [255, 255, 255]], dtype=np.uint8)
        mesh = triangle(colors)
        with tempfile.TemporaryDirectory() as temp:
            path = GlbExporter().export(mesh, Path(temp) / "color.glb")
            document, read = read_glb(path)
            color_index = document["meshes"][0]["primitives"][0]["attributes"]["COLOR_0"]
            actual = read(color_index)
            np.testing.assert_allclose(actual[0], [0, 0.2158605, 1], atol=1e-7)
            self.assertTrue(((actual >= 0) & (actual <= 1)).all())
            np.testing.assert_array_equal(mesh.colors, colors)

    def test_unreferenced_vertices_are_omitted_and_colors_follow_indices(self):
        mesh = triangle(np.array([[255, 0, 0]] * 3, dtype=np.uint8))
        extended = models.Mesh(np.concatenate(([[100, 100, 100]], mesh.vertices)),
                               mesh.faces + 1, np.concatenate(([[0, 255, 0]], mesh.colors)).astype(np.uint8))
        with tempfile.TemporaryDirectory() as temp:
            path = GlbExporter().export(extended, Path(temp) / "compact.glb")
            document, read = read_glb(path)
            attrs = document["meshes"][0]["primitives"][0]["attributes"]
            np.testing.assert_array_equal(read(attrs["POSITION"]), mesh.vertices)
            np.testing.assert_array_equal(read(attrs["COLOR_0"]), [[1, 0, 0]] * 3)

    def test_16_bit_restart_boundary_selects_32_bit_indices(self):
        rng = np.random.default_rng(42)
        with tempfile.TemporaryDirectory() as temp:
            for count, component in ((65535, 5123), (65536, 5125)):
                vertices = rng.random((count, 3))
                faces = np.arange(65535).reshape(-1, 3)
                if count == 65536:
                    faces = np.concatenate((faces, [[65534, 65535, 0]]))
                path = GlbExporter().export(models.Mesh(vertices, faces), Path(temp) / f"{count}.glb")
                document, read = read_glb(path)
                index = document["meshes"][0]["primitives"][0]["indices"]
                self.assertEqual(document["accessors"][index]["componentType"], component)
                np.testing.assert_array_equal(read(index).reshape(-1, 3), faces)

    def test_repeated_export_is_deterministic_and_never_overwrites(self):
        with tempfile.TemporaryDirectory() as temp:
            first, second = Path(temp) / "a.glb", Path(temp) / "b.glb"
            exporter = GlbExporter()
            exporter.export(triangle(), first)
            exporter.export(triangle(), second)
            expected = first.read_bytes()
            self.assertEqual(expected, second.read_bytes())
            with self.assertRaises(FileExistsError):
                exporter.export(triangle(), first)
            self.assertEqual(expected, first.read_bytes())

    def test_invalid_geometry_leaves_no_output(self):
        invalid = [models.Mesh(np.empty((0, 3)), np.empty((0, 3), dtype=int)),
                   models.Mesh(np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]], dtype=float), np.array([[0, 1, 2]])),
                   models.Mesh(np.array([[1e40, 0, 0], [0, 1, 0], [0, 0, 1]]), np.array([[0, 1, 2]])),
                   models.Mesh(np.array([[1e8, 0, 0], [1e8 + 1, 0, 0], [1e8, 1, 0]]), np.array([[0, 1, 2]])),
                   models.Mesh(triangle().vertices, np.array([[0, 1, 2], [2, 1, 0]]))]
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "output.glb"
            for mesh in invalid:
                with self.subTest(mesh=mesh), self.assertRaises(ValueError):
                    GlbExporter().export(mesh, path)
                self.assertFalse(path.exists())
            with self.assertRaises(ValueError):
                GlbExporter().export(triangle(), Path(temp) / "model.gltf")


if __name__ == "__main__":
    unittest.main()
