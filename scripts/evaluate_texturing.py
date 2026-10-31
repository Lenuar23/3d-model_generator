"""Measure vertex color transfer using an analytic colored ellipsoid."""

import argparse
import importlib
import json
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
models = importlib.import_module("3d_model_soft.domain.models")
camera_math = importlib.import_module("3d_model_soft.services.camera_math")
carving = importlib.import_module("3d_model_soft.services.carving")
meshing = importlib.import_module("3d_model_soft.services.meshing")
synthetic = importlib.import_module("3d_model_soft.services.synthetic")
texturing = importlib.import_module("3d_model_soft.services.texturing")
write_png = importlib.import_module("3d_model_soft.infrastructure.synthetic_dataset")._write_rgba_png


def surface_color(points: np.ndarray, radii: np.ndarray) -> np.ndarray:
    return np.rint(np.clip((points / radii + 1) / 2, 0, 1) * 255).astype(np.uint8)


def colored_view(camera, radii):
    """Render RGB from the first analytic ray/ellipsoid intersection."""
    rotation, translation = camera_math.extrinsics(camera)
    origin = -rotation.T @ translation
    v, u = np.indices((camera.height, camera.width), dtype=float)
    focal = camera.height / (2 * np.tan(np.radians(camera.fov_deg) / 2))
    rays = np.stack(((u - (camera.width - 1) / 2) / focal,
                     (v - (camera.height - 1) / 2) / focal, np.ones_like(u)), axis=-1) @ rotation
    scaled_rays, scaled_origin = rays / radii, origin / radii
    a = np.sum(scaled_rays**2, axis=-1)
    b = 2 * np.sum(scaled_rays * scaled_origin, axis=-1)
    c = np.sum(scaled_origin**2) - 1
    depth = (-b - np.sqrt(np.maximum(b**2 - 4 * a * c, 0))) / (2 * a)
    mask = synthetic.ellipsoid_mask(camera, tuple(radii))
    image = np.zeros((camera.height, camera.width, 3), dtype=np.uint8)
    image[mask] = surface_color(origin + rays[mask] * depth[mask, None], radii)
    return models.View(camera, mask, image)


def evaluate(output: Path, resolution: int = 32, size: int = 128) -> dict:
    radii = np.array([0.6, 0.8, 0.5])
    grid = carving.create_grid(resolution=resolution)
    views = [colored_view(cam, radii) for cam in synthetic.zero123_cameras(size=size)]
    mesh = meshing.build_mesh(carving.carve(views, grid))
    started = time.perf_counter()
    result = texturing.texture_mesh(mesh, views, fallback_color=(255, 0, 255))
    elapsed = time.perf_counter() - started
    observed = np.any(result.colors != (255, 0, 255), axis=1)
    errors = np.abs(result.colors[observed].astype(float)
                    - surface_color(result.vertices[observed], radii).astype(float))
    metrics = dict(resolution=resolution, image_size=size, vertices=len(result.vertices),
                   triangles=len(result.faces), observed_vertices=int(observed.sum()),
                   coverage=float(observed.mean()) if len(observed) else 0,
                   mean_absolute_rgb_error=float(errors.mean()) if errors.size else None,
                   texturing_seconds=elapsed,
                   note="Error includes geometry/reprojection error; unseen vertices are magenta.")
    output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(output / "colored_mesh.npz", vertices=result.vertices, faces=result.faces,
                        colors=result.colors, normals=meshing.vertex_normals(result), observed=observed)
    for index, view in enumerate(views):
        rgba = np.concatenate((view.image, (view.mask.astype(np.uint8) * 255)[..., None]), axis=-1)
        write_png(output / f"source_{index:02d}.png", rgba)
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/synthetic/textured"))
    parser.add_argument("--resolution", type=int, default=32)
    parser.add_argument("--size", type=int, default=128)
    args = parser.parse_args()
    try:
        metrics = evaluate(args.output, args.resolution, args.size)
    except (ValueError, FileExistsError) as error:
        parser.error(str(error))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
