"""Evaluate visual hull against analytic objects without AI or image I/O."""

import argparse
import importlib
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
models = importlib.import_module("3d_model_soft.domain.models")
carving = importlib.import_module("3d_model_soft.services.carving")
synthetic = importlib.import_module("3d_model_soft.services.synthetic")


def evaluate(resolution: int = 64, size: int = 256) -> list[dict]:
    """Return IoU, recall and precision for two objects and two angle profiles."""
    grid = carving.create_grid(resolution=resolution)
    indices = np.moveaxis(np.indices(grid.occupancy.shape), 0, -1)
    points = np.asarray(grid.origin) + (indices + 0.5) * grid.spacing
    results = []
    for version in ("v1.1", "v1.2"):
        cameras = synthetic.zero123_cameras(version, size=size)
        for name, radii in (("sphere", (0.65, 0.65, 0.65)), ("ellipsoid", (0.6, 0.8, 0.5))):
            views = [models.View(cam, synthetic.ellipsoid_mask(cam, radii)) for cam in cameras]
            actual = carving.carve(views, grid).occupancy
            truth = synthetic.ellipsoid_occupancy(points, radii)
            intersection = int((actual & truth).sum())
            union = int((actual | truth).sum())
            predicted, expected = int(actual.sum()), int(truth.sum())
            results.append(dict(object=name, profile=version, resolution=resolution,
                                image_size=size, iou=intersection / union if union else 1.0,
                                recall=intersection / expected if expected else None,
                                precision=intersection / predicted if predicted else None,
                                predicted_voxels=predicted, ground_truth_voxels=expected))
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resolution", type=int, default=64)
    parser.add_argument("--size", type=int, default=256)
    args = parser.parse_args()
    try:
        results = evaluate(args.resolution, args.size)
    except ValueError as error:
        parser.error(str(error))
    print(json.dumps(results, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
