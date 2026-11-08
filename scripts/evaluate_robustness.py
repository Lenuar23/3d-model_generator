"""Sweep K-of-N on reproducible synthetic mask failures; no real-data tuning."""

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

SCENARIOS = ("clean", "one_empty", "one_cut", "two_cut", "pixel_noise", "one_shift")


def corrupt_masks(masks: list[np.ndarray], scenario: str, seed: int) -> list[np.ndarray]:
    """Copy masks and apply the named controlled failure without modifying input."""
    result = [mask.copy() for mask in masks]
    height, width = masks[0].shape
    if scenario == "one_empty":
        result[0][:] = False
    elif scenario in ("one_cut", "two_cut"):
        result[0][:, :width // 2] = False
        if scenario == "two_cut":
            result[1][:height // 2] = False
    elif scenario == "pixel_noise":
        rng = np.random.default_rng(seed)
        for target, original in zip(result, masks):
            # Remove 3% of foreground samples, add 1% of background samples.
            target[original & (rng.random(original.shape) < 0.03)] = False
            target[~original & (rng.random(original.shape) < 0.01)] = True
    elif scenario == "one_shift":
        offset = max(1, round(width / 32))
        result[0][:] = False
        result[0][:, offset:] = masks[0][:, :-offset]
    elif scenario != "clean":
        raise ValueError(f"unknown scenario: {scenario}")
    return result


def evaluate(resolution: int = 40, size: int = 128, seed: int = 42) -> dict:
    """Evaluate all K on two objects, two angle profiles and six failure types."""
    if resolution < 8 or size < 16:
        raise ValueError("evaluation requires resolution >= 8 and image size >= 16")
    grid = carving.create_grid(resolution=resolution)
    indices = np.moveaxis(np.indices(grid.occupancy.shape), 0, -1)
    points = np.asarray(grid.origin) + (indices + 0.5) * grid.spacing
    records = []
    for version in ("v1.1", "v1.2"):
        cameras = synthetic.zero123_cameras(version, size=size)
        for name, radii in (("sphere", (0.65, 0.65, 0.65)), ("ellipsoid", (0.6, 0.8, 0.5))):
            masks = [synthetic.ellipsoid_mask(cam, radii) for cam in cameras]
            truth = synthetic.ellipsoid_occupancy(points, radii)
            for scenario in SCENARIOS:
                damaged = corrupt_masks(masks, scenario, seed)
                views = [models.View(camera, mask) for camera, mask in zip(cameras, damaged)]
                for k in range(1, len(views) + 1):
                    actual = carving.carve(views, grid, min_views=k).occupancy
                    intersection, union = int((actual & truth).sum()), int((actual | truth).sum())
                    predicted, expected = int(actual.sum()), int(truth.sum())
                    records.append(dict(object=name, profile=version, scenario=scenario, k=k,
                                        iou=intersection / union if union else 1.0,
                                        recall=intersection / expected if expected else None,
                                        precision=intersection / predicted if predicted else None,
                                        predicted_voxels=predicted, ground_truth_voxels=expected))
    summary = []
    for scenario in SCENARIOS:
        for k in range(1, 7):
            group = [record for record in records if record["scenario"] == scenario and record["k"] == k]
            summary.append(dict(scenario=scenario, k=k,
                                mean_iou=float(np.mean([row["iou"] for row in group])),
                                mean_recall=float(np.mean([row["recall"] for row in group])),
                                mean_precision=(float(np.mean([row["precision"] for row in group]))
                                                if all(row["precision"] is not None for row in group) else None)))
    return dict(resolution=resolution, image_size=size, seed=seed, source="synthetic",
                note="No real-view calibration. Means cover two objects and two angle profiles.",
                summary=summary, records=records)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/synthetic/robustness"))
    parser.add_argument("--resolution", type=int, default=40)
    parser.add_argument("--size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output directory already exists; choose a new --output")
    try:
        report = evaluate(args.resolution, args.size, args.seed)
    except ValueError as error:
        parser.error(str(error))
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "metrics.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    for row in report["summary"]:
        if row["k"] >= 4:
            print(f"{row['scenario']:12s} K={row['k']}  IoU={row['mean_iou']:.4f}  recall={row['mean_recall']:.4f}")
    print(f"Saved all thresholds and per-scene metrics: {args.output.resolve() / 'metrics.json'}")


if __name__ == "__main__":
    main()
