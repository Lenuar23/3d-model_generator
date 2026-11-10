"""Demonstrate full 3D reconstruction with enhanced lighting, normals and materials.

Usage:
    .venv\\Scripts\\python.exe scripts/demo_reconstruct.py --object duck --resolution 80 --smoothing 6
"""

import argparse
import importlib
import json
from pathlib import Path
import struct
import sys
import time

if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
models = importlib.import_module("3d_model_soft.domain.models")
carving = importlib.import_module("3d_model_soft.services.carving")
meshing = importlib.import_module("3d_model_soft.services.meshing")
texturing = importlib.import_module("3d_model_soft.services.texturing")

ALL_OBJECTS = {
    "apple": ("data/task5_validation/2f6b318023484aef9ec0a105091b1ba5/views", "01_simple_apple.jpg"),
    "tennis_ball": ("data/task5_validation/ca3ea6fc84cb419fa6c86925eb3e8806/views", "02_simple_tennis_ball.jpg"),
    "soap": ("data/task5_validation/4395616a716d4926b362cd6f9f411e47/views", "03_simple_soap.jpg"),
    "pine_cone": ("data/task5_validation/d5c449ecd8f74ea6a3e5e65e5e6d5295/views", "04_complex_pine_cone.jpg"),
    "duck": ("data/task5_validation/0278fa96a6c144e9a8e801699258e920/views", "05_complex_rubber_duck.jpg"),
    "padlock": ("data/task5_validation/0d7f2cf3d31445498f1baf968fb629f2/views", "06_complex_padlock.jpg"),
    "spoon": ("data/task5_validation/4d32ca9b314a4c0f88c64e152108b06e/views", "07_shiny_spoon.jpg"),
    "teapot": ("data/task5_validation/e3485c28282b4d4fb088e6d05f20e290/views", "08_shiny_silver_teapot.jpg"),
    "bottle": ("data/task5_validation/d7fa70abff63426da738bbb3461ccca0/views", "09_transparent_bottle.jpg"),
    "goblet": ("data/task5_validation/df18d570ba73414db459efa6e556aa5a/views", "10_transparent_goblet.jpg"),
}


def export_glb(mesh, filepath: Path) -> Path:
    """Export high quality glTF 2.0 binary with smooth normals, vertex colors, and PBR material."""
    verts = mesh.vertices.astype(np.float32)
    faces = mesh.faces.astype(np.uint32)
    normals = meshing.vertex_normals(mesh).astype(np.float32)
    has_color = mesh.colors is not None

    vert_bytes = verts.tobytes()
    normal_bytes = normals.tobytes()
    face_bytes = faces.flatten().tobytes()
    bin_data = vert_bytes + normal_bytes + face_bytes

    accessors = [
        # 0: POSITION
        {
            "bufferView": 0,
            "byteOffset": 0,
            "componentType": 5126,
            "count": len(verts),
            "type": "VEC3",
            "max": verts.max(axis=0).tolist(),
            "min": verts.min(axis=0).tolist(),
        },
        # 1: NORMAL (smooth vertex normals make 3D model look significantly better)
        {
            "bufferView": 1,
            "byteOffset": 0,
            "componentType": 5126,
            "count": len(normals),
            "type": "VEC3",
        },
        # 2: INDICES
        {
            "bufferView": 2,
            "byteOffset": 0,
            "componentType": 5125,
            "count": faces.size,
            "type": "SCALAR",
        },
    ]

    buffer_views = [
        # 0: Position
        {
            "buffer": 0,
            "byteOffset": 0,
            "byteLength": len(vert_bytes),
            "target": 34962,
        },
        # 1: Normal
        {
            "buffer": 0,
            "byteOffset": len(vert_bytes),
            "byteLength": len(normal_bytes),
            "target": 34962,
        },
        # 2: Face indices
        {
            "buffer": 0,
            "byteOffset": len(vert_bytes) + len(normal_bytes),
            "byteLength": len(face_bytes),
            "target": 34963,
        },
    ]

    attributes = {"POSITION": 0, "NORMAL": 1}

    if has_color:
        colors = mesh.colors.astype(np.uint8)
        color_bytes = colors.tobytes()
        pad = (4 - (len(bin_data) % 4)) % 4
        bin_data += b"\x00" * pad
        color_offset = len(bin_data)
        bin_data += color_bytes

        accessors.append({
            "bufferView": len(buffer_views),
            "byteOffset": 0,
            "componentType": 5121,
            "count": len(colors),
            "type": "VEC3",
            "normalized": True,
        })
        buffer_views.append({
            "buffer": 0,
            "byteOffset": color_offset,
            "byteLength": len(color_bytes),
            "target": 34962,
        })
        attributes["COLOR_0"] = len(accessors) - 1

    pad = (4 - (len(bin_data) % 4)) % 4
    bin_data += b"\x00" * pad

    # PBR material definition for natural non-plastic diffuse look
    materials = [
        {
            "name": "ObjectMaterial",
            "pbrMetallicRoughness": {
                "baseColorFactor": [1.0, 1.0, 1.0, 1.0],
                "metallicFactor": 0.05,
                "roughnessFactor": 0.65,
            },
            "doubleSided": True,
        }
    ]

    gltf = {
        "asset": {"version": "2.0", "generator": "3d-model-soft-enhanced"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0}],
        "meshes": [
            {
                "primitives": [
                    {
                        "attributes": attributes,
                        "indices": 2,
                        "material": 0,
                    }
                ]
            }
        ],
        "materials": materials,
        "accessors": accessors,
        "bufferViews": buffer_views,
        "buffers": [{"byteLength": len(bin_data)}],
    }

    json_bytes = json.dumps(gltf).encode("utf-8")
    pad_json = (4 - (len(json_bytes) % 4)) % 4
    json_bytes += b" " * pad_json

    total_size = 12 + 8 + len(json_bytes) + 8 + len(bin_data)
    header = struct.pack("<4sII", b"glTF", 2, total_size)
    json_chunk_hdr = struct.pack("<II", len(json_bytes), 0x4E4F534A)
    bin_chunk_hdr = struct.pack("<II", len(bin_data), 0x004E4942)

    filepath.parent.mkdir(parents=True, exist_ok=True)
    with filepath.open("wb") as f:
        f.write(header)
        f.write(json_chunk_hdr)
        f.write(json_bytes)
        f.write(bin_chunk_hdr)
        f.write(bin_data)
    return filepath


def export_ply(mesh, filepath: Path) -> Path:
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with filepath.open("w", encoding="utf-8") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(mesh.vertices)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        if mesh.colors is not None:
            f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write(f"element face {len(mesh.faces)}\n")
        f.write("property list uchar int vertex_indices\nend_header\n")
        for i, v in enumerate(mesh.vertices):
            if mesh.colors is not None:
                c = mesh.colors[i]
                f.write(f"{v[0]:.4f} {v[1]:.4f} {v[2]:.4f} {int(c[0])} {int(c[1])} {int(c[2])}\n")
            else:
                f.write(f"{v[0]:.4f} {v[1]:.4f} {v[2]:.4f}\n")
        for face in mesh.faces:
            f.write(f"3 {face[0]} {face[1]} {face[2]}\n")
    return filepath


def load_views(views_dir: Path, alpha_threshold: int = 128):
    cam_data = json.loads((views_dir / "cameras.json").read_text(encoding="utf-8"))
    loaded = []
    for name, params in sorted(cam_data.items()):
        img = Image.open(views_dir / name)
        rgba = np.array(img)
        cam = models.Camera(
            float(params["azimuth"]),
            float(params["elevation"]),
            float(params["distance"]),
            float(params["fov"]),
            img.width,
            img.height,
        )
        loaded.append(models.View(cam, rgba[..., 3] >= alpha_threshold, rgba[..., :3].copy()))
    return loaded


def reconstruct_3d(views_dir: Path, output_dir: Path, resolution: int = 64, smoothing: int = 5):
    start = time.perf_counter()
    print(f"[*] Завантаження ракурсів з: {views_dir}", flush=True)
    views = load_views(views_dir)
    print(f"    Завантажено {len(views)} ракурсів ({views[0].camera.width}x{views[0].camera.height})", flush=True)

    print(f"[*] Крок 1: Space Carving (воксельна сітка {resolution}x{resolution}x{resolution})...", flush=True)
    grid = carving.create_grid(resolution=resolution)
    carved = carving.carve(views, grid)
    print(f"    Заповнено вокселів: {int(carved.occupancy.sum())} з {resolution**3}", flush=True)

    print("[*] Крок 2: Marching Cubes (побудова полігональної сітки)...", flush=True)
    raw_mesh = meshing.clean_mesh(meshing.marching_cubes(carved))
    print(f"    Побудовано: {len(raw_mesh.vertices)} вершин, {len(raw_mesh.faces)} граней", flush=True)

    print(f"[*] Крок 3: Taubin smoothing ({smoothing} ітерацій без стискання об'єму)...", flush=True)
    smoothed = meshing.smooth_mesh(raw_mesh, iterations=smoothing)

    print("[*] Крок 4: Текстурування вершин (Vertex coloring з 6 ракурсів)...", flush=True)
    t0 = time.perf_counter()
    colored = texturing.texture_mesh(smoothed, views)
    print(f"    Текстурування виконано за {time.perf_counter() - t0:.2f} с", flush=True)

    print("[*] Крок 5: Збереження (GLB з нормалями та PBR + PLY)...", flush=True)
    glb_path = export_glb(colored, output_dir / "model.glb")
    ply_path = export_ply(colored, output_dir / "model.ply")
    elapsed = time.perf_counter() - start

    print("\n" + "=" * 50, flush=True)
    print(f"[+] РЕКОНСТРУЙОВАНО за {elapsed:.2f} сек!", flush=True)
    print(f"    - GLB файл (з нормалями): {glb_path.resolve()}", flush=True)
    print(f"    - Вершин: {len(colored.vertices)}, Трикутників: {len(colored.faces)}", flush=True)
    print("=" * 50, flush=True)
    return glb_path


def main():
    parser = argparse.ArgumentParser(description="3D Model Reconstruction Demo")
    parser.add_argument("--object", choices=list(ALL_OBJECTS.keys()), default="apple",
                        help="Вибір тестового об'єкта зі списку")
    parser.add_argument("--views-dir", type=Path, default=None,
                        help="Шлях до папки з view_00..view_05.png та cameras.json")
    parser.add_argument("--resolution", type=int, default=64,
                        help="Роздільність воксельної сітки (рекомендовано 64 або 80)")
    parser.add_argument("--smoothing", type=int, default=5,
                        help="Кількість ітерацій згладжування Таубіна (рекомендовано 5)")
    parser.add_argument("--output", type=Path, default=Path("data/demo_output"),
                        help="Вихідна папка")
    args = parser.parse_args()

    views_path = args.views_dir if args.views_dir is not None else Path(ALL_OBJECTS[args.object][0])
    if not views_path.exists():
        parser.error(f"Папка ракурсів не знайдена: {views_path}")

    reconstruct_3d(views_path, args.output, resolution=args.resolution, smoothing=args.smoothing)


if __name__ == "__main__":
    main()
