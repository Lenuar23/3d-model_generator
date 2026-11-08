"""Self-contained glTF 2.0 binary export with vertex colors; no runtime SDK."""

import json
from pathlib import Path
import struct

import numpy as np

from ...domain.models import Mesh


class GlbExporter:
    """Implement MeshExporter with POSITION, NORMAL and optional COLOR_0.

    Coordinates stay in the project's right-handed +Y-up world convention.
    Mesh colors are sRGB bytes; glTF vertex colors are linear float values.
    Unreferenced vertices are omitted. Invalid/degenerate triangles are rejected.
    Existing output files are never overwritten.
    """

    def export(self, mesh: Mesh, output_path: str | Path) -> Path:
        """Write an embedded GLB 2.0 scene, creating parent directories as needed."""
        path = Path(output_path)
        if path.suffix.lower() != ".glb":
            raise ValueError("output_path must have a .glb extension")
        if not len(mesh.faces):
            raise ValueError("cannot export a mesh without triangles")
        used, remapped = np.unique(mesh.faces, return_inverse=True)
        faces = remapped.reshape(-1, 3)
        with np.errstate(over="ignore", invalid="ignore"):
            positions = mesh.vertices[used].astype("<f4")
        if not np.isfinite(positions).all():
            raise ValueError("vertex coordinates must fit finite float32 values")
        # Validate the actual float32 geometry the consumer will read.
        triangles = positions.astype(float)[faces]
        face_normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
        if (np.linalg.norm(face_normals, axis=1) == 0).any():
            raise ValueError("mesh contains triangles degenerate at float32 precision")
        normals = np.zeros((len(positions), 3), dtype=float)
        for corner in range(3):
            np.add.at(normals, faces[:, corner], face_normals)
        lengths = np.linalg.norm(normals, axis=1)
        if (lengths == 0).any():
            raise ValueError("mesh has undefined vertex normals; check face winding")
        normals = (normals / lengths[:, None]).astype("<f4")

        payload = bytearray()
        buffer_views, accessors = [], []

        def add_array(array: np.ndarray, component: int, kind: str, target: int,
                      *, bounds: bool = False) -> int:
            payload.extend(b"\x00" * (-len(payload) % 4))
            offset = len(payload)
            raw = array.tobytes(order="C")
            payload.extend(raw)
            view_index = len(buffer_views)
            buffer_views.append(dict(buffer=0, byteOffset=offset, byteLength=len(raw), target=target))
            accessor = dict(bufferView=view_index, byteOffset=0, componentType=component,
                            count=len(array), type=kind)
            if bounds:
                accessor["min"] = array.min(axis=0).tolist()
                accessor["max"] = array.max(axis=0).tolist()
            accessors.append(accessor)
            return len(accessors) - 1

        attributes = dict(POSITION=add_array(positions, 5126, "VEC3", 34962, bounds=True),
                          NORMAL=add_array(normals, 5126, "VEC3", 34962))
        if mesh.colors is not None:
            srgb = mesh.colors[used].astype(float) / 255
            linear = np.where(srgb <= 0.04045, srgb / 12.92, ((srgb + 0.055) / 1.055)**2.4)
            attributes["COLOR_0"] = add_array(linear.astype("<f4"), 5126, "VEC3", 34962)
        # glTF forbids the maximum representable index (primitive restart).
        short_indices = len(positions) <= 65535
        indices = faces.reshape(-1).astype("<u2" if short_indices else "<u4")
        index_accessor = add_array(indices, 5123 if short_indices else 5125, "SCALAR", 34963)
        material = dict(pbrMetallicRoughness=dict(baseColorFactor=[1.0, 1.0, 1.0, 1.0],
                                                metallicFactor=0.0, roughnessFactor=1.0))
        document = dict(asset=dict(version="2.0", generator="3D Model Soft"), scene=0,
                        scenes=[dict(nodes=[0])], nodes=[dict(mesh=0)],
                        meshes=[dict(primitives=[dict(attributes=attributes, indices=index_accessor,
                                                      material=0, mode=4)])],
                        materials=[material], buffers=[dict(byteLength=len(payload))],
                        bufferViews=buffer_views, accessors=accessors)
        encoded = json.dumps(document, separators=(",", ":"), allow_nan=False).encode("utf-8")
        encoded += b" " * (-len(encoded) % 4)
        payload.extend(b"\x00" * (-len(payload) % 4))
        total_length = 12 + 8 + len(encoded) + 8 + len(payload)
        if total_length > 0xFFFFFFFF:
            raise ValueError("GLB exceeds the 32-bit file-size limit")
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as output:
            output.write(struct.pack("<4sII", b"glTF", 2, total_length))
            output.write(struct.pack("<I4s", len(encoded), b"JSON"))
            output.write(encoded)
            output.write(struct.pack("<I4s", len(payload), b"BIN\x00"))
            output.write(payload)
        return path
