"""Run silhouettes -> carving -> mesh and save geometry for inspection."""

import argparse
import importlib
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
models = importlib.import_module("3d_model_soft.domain.models")
carving = importlib.import_module("3d_model_soft.services.carving")
meshing = importlib.import_module("3d_model_soft.services.meshing")
synthetic = importlib.import_module("3d_model_soft.services.synthetic")


def evaluate(output: Path, resolution: int = 48, size: int = 256) -> list[dict]:
    """Save raw/smoothed meshes and report signed volume and triangle counts."""
    grid = carving.create_grid(resolution=resolution)
    results, meshes = [], []
    for version in ("v1.1", "v1.2"):
        cameras = synthetic.zero123_cameras(version, size=size)
        for name, radii in (("sphere", (0.65, 0.65, 0.65)), ("ellipsoid", (0.6, 0.8, 0.5))):
            views = [models.View(cam, synthetic.ellipsoid_mask(cam, radii)) for cam in cameras]
            voxels = carving.carve(views, grid)
            raw = meshing.clean_mesh(meshing.marching_cubes(voxels))
            smooth = meshing.smooth_mesh(raw)
            volumes = []
            for mesh in (raw, smooth):
                a, b, c = np.moveaxis(mesh.vertices[mesh.faces], 1, 0)
                volumes.append(float(np.einsum("ij,ij->", a, np.cross(b, c)) / 6))
            results.append(dict(object=name, profile=version, resolution=resolution,
                                image_size=size, vertices=len(smooth.vertices), triangles=len(smooth.faces),
                                raw_volume=volumes[0], smoothed_volume=volumes[1],
                                analytic_volume=float(4 / 3 * np.pi * np.prod(radii))))
            meshes.append((f"{name}_{version}.npz", raw, smooth))
    output.mkdir(parents=True, exist_ok=False)
    for name, raw, smooth in meshes:
        np.savez_compressed(output / name, vertices=smooth.vertices, faces=smooth.faces,
                            normals=meshing.vertex_normals(smooth), raw_vertices=raw.vertices)
    (output / "metrics.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/synthetic/meshes"))
    parser.add_argument("--resolution", type=int, default=48)
    parser.add_argument("--size", type=int, default=256)
    args = parser.parse_args()
    try:
        results = evaluate(args.output, args.resolution, args.size)
    except (ValueError, FileExistsError) as error:
        parser.error(str(error))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
