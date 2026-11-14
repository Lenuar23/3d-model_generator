"""Orchestrate the reconstruction without a dependency on any UI state."""
from dataclasses import asdict, dataclass
from collections.abc import Callable
from pathlib import Path
from time import perf_counter
import json
from tempfile import mkdtemp

import numpy as np
from PIL import Image

from . import diagnostics
from .exporter import Exports, export_model
from .image_processing import Photo, load_photo
from .mesh_generator import Mesh, generate, validate
from .segmentation import Segmentation, segment
from .silhouette import Profile, analyze
from .texture_mapper import project


@dataclass(frozen=True)
class Settings:
    threshold: float = 0
    background: str | None = None
    contrast: float = 1
    angular: int = 96
    vertical: int = 96
    smoothing: float = 0.018
    radius_factor: float = 1
    scale: float = 1
    close_bottom: bool = True
    close_top: bool = True
    symmetry: bool = False
    mirrored: bool = False
    texture_size: int = 512
    shadow_strength: float = 0.6
    mask_smoothing: float = 1
    rounded: bool = False

    def validate(self) -> None:
        """Keep sampling, numeric settings and intermediate memory within bounds."""
        for name, lower, upper in (
            ("threshold", 0, 100), ("contrast", 0.5, 2), ("angular", 8, 192),
            ("vertical", 8, 192), ("smoothing", 0, 0.12), ("radius_factor", 0.25, 2),
            ("scale", 0.01, 10),
            ("texture_size", 64, 1024),
            ("shadow_strength", 0, 1), ("mask_smoothing", 0, 2),
        ):
            if not lower <= getattr(self, name) <= upper:
                raise ValueError(f"{name} must be between {lower} and {upper}.")
        if any(int(value) != value for value in (self.angular, self.vertical, self.texture_size)):
            raise ValueError("Angular samples, height samples and texture size must be integers.")


@dataclass
class Result:
    exports: Exports
    report: dict
    images: dict[str, np.ndarray]
    profile: Profile
    mesh: Mesh
    atlas: np.ndarray


def make_mask(photo: Photo, settings: Settings, strokes: Image.Image | None = None) -> Segmentation:
    """Apply the same foreground options for preview and reconstruction."""
    add = remove = None
    if strokes is not None:
        if not isinstance(strokes, Image.Image) or strokes.size != photo.rgb.shape[1::-1]:
            raise ValueError("Correction canvas does not match the photo. Reset corrections and repaint.")
        rgba = np.asarray(strokes.convert("RGBA"))
        alpha = rgba[..., 3] >= 100
        add = alpha & (rgba[..., 1] > rgba[..., 0])
        remove = alpha & (rgba[..., 0] > rgba[..., 1])
    return segment(photo, settings.threshold, settings.background, add, remove,
                   settings.shadow_strength, settings.mask_smoothing)


def reconstruct(
    source: str | Path | Image.Image, settings: Settings = Settings(),
    output_root: Path = Path("outputs"), strokes: Image.Image | None = None,
    progress: Callable[[float, str], None] | None = None,
) -> Result:
    """Segment, revolve, texture, validate and export one independent run."""
    start = perf_counter()
    settings.validate()
    timings = {}
    checkpoint = start

    def mark(name: str) -> None:
        nonlocal checkpoint
        now = perf_counter()
        timings[name] = round(now - checkpoint, 5)
        checkpoint = now

    if progress is not None:
        progress(0.1, "Preparing image and segmenting foreground")
    photo = load_photo(source, contrast=settings.contrast)
    mark("preprocessing")
    region = make_mask(photo, settings, strokes)
    mark("segmentation")
    profile = analyze(region.mask, int(settings.vertical), settings.smoothing,
                      settings.symmetry, settings.rounded)
    mark("silhouette")
    if progress is not None:
        progress(0.35, "Building and checking the mesh")
    mesh_start = perf_counter()
    mesh = generate(profile.heights * settings.scale,
                    profile.sample_radii * settings.radius_factor * settings.scale,
                    int(settings.angular), settings.close_bottom, settings.close_top)
    mesh_seconds = perf_counter() - mesh_start
    mark("mesh_generation")
    checks = validate(mesh)
    expected_boundary = (2 - len(mesh.cap_ends)) * settings.angular
    if (checks["degenerate_triangles"] or checks["nonmanifold_edges"] or checks["inconsistent_edges"]
        or checks["boundary_edges"] != expected_boundary or not checks["unit_normals"]
        or not checks["finite_vertices"] or checks["duplicate_faces"] or checks["isolated_vertices"]
        or checks["connected_components"] != 1 or checks["opposed_face_normals"]):
        raise ValueError("Mesh validation failed. Try less smoothing or a different silhouette.")
    mark("mesh_validation")
    if progress is not None:
        progress(0.6, "Baking the texture atlas")
    texture = project(photo.rgb, region.mask, profile, mesh, settings.mirrored, int(settings.texture_size))
    mark("texture")
    frames = diagnostics.images(photo.rgb, region)
    output_root.mkdir(parents=True, exist_ok=True)
    directory = Path(mkdtemp(prefix="run-", dir=output_root))
    if progress is not None:
        progress(0.8, "Writing GLB, OBJ and diagnostics")
    exports = export_model(texture, directory)
    mark("export")
    for name, frame in {**frames, "atlas": texture.pixels}.items():
        Image.fromarray(frame).save(directory / f"{name}.png", compress_level=1)
    diagnostics.radius_plot(profile, settings.radius_factor).write_html(
        directory / "radius.html", include_plotlyjs=True,
    )
    mark("diagnostics")
    _, _, width, height = region.bbox
    report = {
        "input_resolution": list(photo.original_size),
        "working_resolution": list(photo.rgb.shape[1::-1]),
        "bounding_box_xywh": list(region.bbox),
        "segmentation_coverage_percent": round(100 * region.mask.mean(), 2),
        "height_to_width": round(height / width, 3),
        "color_threshold_used": region.threshold,
        "segmentation_indicators": region.indicators,
        "silhouette_indicators": profile.indicators,
        "texture_indicators": {
            "seam_max_rgb_difference": texture.seam_difference,
            "rear_max_row_rgb_jump": texture.rear_row_jump,
        },
        "vertices": len(mesh.vertices),
        "triangles": len(mesh.faces),
        "export_vertices_with_uv_splits": len(texture.vertices),
        "mesh_generation_seconds": round(mesh_seconds, 5),
        "total_processing_seconds": round(perf_counter() - start, 4),
        "stage_seconds": timings,
        "parameters": asdict(settings),
        "warnings": region.warnings + profile.warnings,
        "mesh_checks": checks,
    }
    (directory / "diagnostics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return Result(exports, report, frames, profile, mesh, texture.pixels)
