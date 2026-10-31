"""Strict visual hull: keep voxel centers inside every calibrated silhouette."""

from collections.abc import Sequence
from numbers import Integral

import numpy as np

from ..domain.models import View, VoxelGrid
from .camera_math import project


def create_grid(
    bounds_min: tuple[float, float, float] = (-1.0, -1.0, -1.0),
    bounds_max: tuple[float, float, float] = (1.0, 1.0, 1.0),
    resolution: int | tuple[int, int, int] = 64,
) -> VoxelGrid:
    """Create fully occupied cells spanning an axis-aligned world-space box."""
    lower = np.asarray(bounds_min, dtype=float)
    upper = np.asarray(bounds_max, dtype=float)
    if (lower.shape != (3,) or upper.shape != (3,)
            or not np.isfinite(lower).all() or not np.isfinite(upper).all()
            or (upper <= lower).any()):
        raise ValueError("bounds must be finite 3D corners with max > min on every axis")
    try:
        shape = (resolution,) * 3 if isinstance(resolution, Integral) else tuple(resolution)
    except TypeError as error:
        raise ValueError("resolution must be an integer or a sequence of three integers") from error
    if (len(shape) != 3 or any(isinstance(n, bool) or not isinstance(n, Integral)
                              or n < 1 for n in shape)):
        raise ValueError("resolution must be a positive integer or three positive integers")
    spacing = (upper - lower) / np.asarray(shape)
    if not np.isfinite(spacing).all():
        raise ValueError("bounds produce nonfinite spacing")
    return VoxelGrid(np.ones(shape, dtype=bool), tuple(lower), tuple(spacing))


def carve(views: Sequence[View], grid: VoxelGrid, *, chunk_size: int = 65536) -> VoxelGrid:
    """Return a carved copy; never change input occupancy or masks.

    Project cell centers and sample the nearest pixel (floor(pixel + 0.5)).
    The image footprint is [-0.5, width-0.5) x [-0.5, height-0.5).
    Nonfinite, behind-camera and out-of-frame points are carved away.
    All views must contain the object; cropped/unreliable views can over-carve.
    Work is vectorized in chunks to bound temporary coordinate-array memory.
    """
    if not views:
        raise ValueError("at least one view is required")
    if (isinstance(chunk_size, bool) or not isinstance(chunk_size, Integral)
            or chunk_size < 1):
        raise ValueError("chunk_size must be a positive integer")
    if any(not isinstance(view, View) for view in views):
        raise ValueError("views must contain View objects")
    result = grid.occupancy.copy()
    flat = result.reshape(-1)
    for start in range(0, flat.size, chunk_size):
        stop = min(start + chunk_size, flat.size)
        active = np.flatnonzero(flat[start:stop]) + start
        if not active.size:
            continue
        indices = np.column_stack(np.unravel_index(active, result.shape))
        points = np.asarray(grid.origin) + (indices + 0.5) * np.asarray(grid.spacing)
        keep = np.ones(len(active), dtype=bool)
        for view in views:
            candidates = np.flatnonzero(keep)
            if not candidates.size:
                break
            pixels = project(points[candidates], view.camera)
            in_frame = (np.isfinite(pixels).all(axis=1)
                        & (pixels[:, 0] >= -0.5) & (pixels[:, 0] < view.camera.width - 0.5)
                        & (pixels[:, 1] >= -0.5) & (pixels[:, 1] < view.camera.height - 0.5))
            hits = np.zeros(len(candidates), dtype=bool)
            uv = np.floor(pixels[in_frame] + 0.5).astype(np.intp)
            hits[in_frame] = view.mask[uv[:, 1], uv[:, 0]]
            keep[candidates] = hits
        flat[active] = keep
    return VoxelGrid(result, grid.origin, grid.spacing)
