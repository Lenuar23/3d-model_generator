"""Prepare real or inferred views, reconstruct, and export the resulting visual hull."""
from dataclasses import dataclass
from pathlib import Path
from tempfile import mkdtemp
from time import perf_counter
import json

import numpy as np
from PIL import Image

from .camera import Camera
from .ai_assist import read_generated_manifest
from .exporter import Exports, export_model
from .image_processing import load_photo
from .pipeline import Settings, make_mask
from .visual_hull import Hull, View, carve, consistency
from .multiview_texture import bake
from .mesh_generator import validate


def prepare(paths: list[str], rows: list[list], inferred: bool = False) -> tuple[list[View], list[str]]:
    """Use enabled rows with explicit camera centers/scales in the oriented original image."""
    if len(paths) != len(rows) or not 3 <= len(paths) <= 8:
        raise ValueError("Supply 3–8 images and one camera row per image.")
    views, warnings = [], []
    for index, (path, row) in enumerate(zip(paths, rows), 1):
        angle, elevation, scale, cx, cy, enabled = row
        if inferred and index == 1 and not enabled:
            raise ValueError("Keep the observed photograph enabled in AI-Assisted mode.")
        if not enabled:
            continue
        photo = load_photo(path)
        working_height, working_width = photo.alpha.shape
        sx, sy = np.array([working_width, working_height]) / photo.original_size
        camera = Camera(float(angle), float(elevation), float(scale)*sx,
                        (float(cx)+0.5)*sx-0.5, (float(cy)+0.5)*sy-0.5)
        if abs(sx - sy) / sx > 0.05:
            raise ValueError("Image resizing would distort this camera. Use a less elongated image.")
        region = make_mask(photo, Settings(shadow_strength=0))
        name = f"View {index}: {Path(path).name}"
        warnings.extend(f"{name}: {warning}" for warning in region.warnings)
        views.append(View(photo.rgb, region.mask, camera, name, inferred and index != 1))
    return views, warnings


@dataclass
class MultiResult:
    exports: Exports
    hull: Hull
    atlas: np.ndarray
    report: dict
    provenance: Path


class HullValidationError(ValueError):
    """The reconstructed surface failed topology checks."""


def reconstruct_views(views: list[View], resolution: int = 48, extent: float = 1.6,
                      output_root: Path = Path("outputs"), progress=None,
                      preparation_warnings: list[str] | None = None, prior=None,
                      texture_consensus: bool = False, normal_observations=(),
                      normal_strength: float = .65, normal_basis=(1, 1, 1)) -> MultiResult:
    """Run the same deterministic geometry engine for either source of views."""
    start = perf_counter()
    warnings = list(dict.fromkeys([*(preparation_warnings or []), *consistency(views)]))
    if progress:
        progress(0.2, "Carving silhouettes")
    polyhedral = None
    if prior is None and normal_observations and any(view.inferred for view in views):
        from .hull_quality import polyhedral_prior

        polyhedral = polyhedral_prior(normal_observations[0][0], normal_observations[0][1], normal_basis)
    hull = carve(views, resolution, extent, prior=prior, polyhedral=polyhedral)
    initial = hull
    normal_report = None
    if normal_observations:
        from .surface_refinement import refine, targets

        desired, confidence, target_info = targets(
            hull, normal_observations, normal_basis, return_info=True, auto_basis=True
        )
        hull, normal_report = refine(
            hull, desired, confidence, views, normal_strength, basis_used=target_info["basis_used"]
        )
        if not normal_report["accepted"]:
            warnings.append("Normal refinement retained the initial surface. "+normal_report["reason"])
    checks = validate(hull)
    if (not checks["watertight"] or not checks["finite_vertices"] or not checks["unit_normals"]
        or checks["connected_components"] != 1 or checks["degenerate_triangles"]
        or checks["inconsistent_edges"] or checks["isolated_vertices"]):
        raise HullValidationError("Hull mesh validation failed. Check the input masks or change voxel resolution.")
    carved = perf_counter()
    if hull.diagnostics["volume_boundary_touched"]:
        warnings.append("The hull reaches the volume boundary. Increase the volume or check calibration.")
    if any(view.inferred for view in views):
        warnings.append("Generated views are inferred. Their poses and silhouettes can bias geometry.")
    if progress:
        progress(0.55, "Projecting visible colors")
    texture, texture_report, provenance = bake(hull, views, consensus=texture_consensus)
    textured = perf_counter()
    if progress:
        progress(0.85, "Exporting the mesh")
    output_root.mkdir(parents=True, exist_ok=True)
    directory = Path(mkdtemp(prefix="hull-", dir=output_root))
    exports = export_model(texture, directory)
    if normal_report is not None:
        import trimesh
        import plotly.graph_objects as go

        mesh = trimesh.Trimesh(initial.vertices, initial.faces, process=False)
        mesh.export(directory / "initial.glb")
        values = [normal_report.get("objective_before"), *[trial["objective"]
                  for trial in normal_report.get("trials", [])]]
        go.Figure(go.Scatter(y=values, mode="lines+markers")).update_layout(
            xaxis_title="Backtracking trial", yaxis_title="Tangent and regularization objective"
        ).write_html(directory / "normal-objective.html")
    for index, view in enumerate(views, 1):
        Image.fromarray(view.rgb).save(directory / f"view-{index}.png")
        Image.fromarray(view.mask).save(directory / f"mask-{index}.png")
    Image.fromarray(texture.pixels).save(directory / "atlas.png")
    provenance_path = directory / "texture-provenance.png"
    Image.fromarray(provenance).save(provenance_path)
    report = {
        **hull.diagnostics, **texture_report,
        "mesh_checks": checks,
        "vertices": len(hull.vertices), "triangles": len(hull.faces), "warnings": warnings,
        "carving_seconds": round(carved-start, 4), "texture_seconds": round(textured-carved, 4),
        "elapsed_seconds": round(perf_counter()-start, 4), "voxel_resolution": resolution,
        "occupancy_grid_bytes": hull.occupancy.nbytes,
        "polyhedral_prior_applied": bool(polyhedral is not None),
        "cameras": [{**vars(view.camera), "name": view.name, "inferred": view.inferred}
                    for view in views],
    }
    if normal_report is not None:
        report["normal_refinement"] = normal_report
    (directory / "diagnostics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return MultiResult(exports, hull, texture.pixels, report, provenance_path)


def main() -> None:
    """Run the photo or generated-view workflow from explicit camera metadata."""
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cameras", type=Path, required=True, help="JSON array of camera rows.")
    parser.add_argument("--generated-manifest", type=Path, help="Reviewed generated-view manifest.")
    parser.add_argument("--resolution", type=int, default=48)
    parser.add_argument("--extent", type=float, default=1.6)
    parser.add_argument("--output", type=Path, default=Path("outputs"))
    parser.add_argument("images", nargs="+")
    args = parser.parse_args()
    try:
        rows = json.loads(args.cameras.read_text(encoding="utf-8"))
        paths = args.images
        if args.generated_manifest:
            if len(paths) != 1 or len(rows) != 1:
                raise ValueError("Provide exactly one observed image and one observed camera row.")
            for entry in read_generated_manifest(args.generated_manifest):
                paths.append(str(entry.path))
                camera = entry.camera
                rows.append([camera.azimuth, camera.elevation, camera.pixels_per_unit,
                             camera.cx, camera.cy, True])
        views, warnings = prepare(paths, rows, inferred=bool(args.generated_manifest))
        result = reconstruct_views(views, args.resolution, args.extent, args.output,
                                   preparation_warnings=warnings)
        if args.generated_manifest:
            (result.exports.glb.parent / "generated-manifest.json").write_text(
                args.generated_manifest.read_text(encoding="utf-8"), encoding="utf-8")
    except (OSError, ValueError, TypeError) as error:
        parser.error(str(error))
    print(result.exports.glb.resolve().parent)


if __name__ == "__main__":
    main()

