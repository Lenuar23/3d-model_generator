"""Analytic ellipsoid silhouettes and volume samples for reconstruction tests."""

import numpy as np

from ..domain.models import Camera
from .camera_math import extrinsics


def _radii_array(radii: tuple[float, float, float]) -> np.ndarray:
    values = np.asarray(radii, dtype=np.float64)
    if values.shape != (3,) or not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError("radii must contain three finite positive values")
    return values


def zero123_cameras(
    version: str = "v1.2", *, size: int = 256, distance: float = 4.0,
    fov_deg: float = 30.0,
) -> tuple[Camera, ...]:
    """Use published Zero123++ angles; distance/FOV are synthetic settings."""
    elevations = {"v1.1": (30, -20), "v1.2": (20, -10)}
    if version not in elevations:
        raise ValueError("version must be v1.1 or v1.2")
    return tuple(
        Camera(azimuth, elevations[version][index % 2], distance, fov_deg, size, size)
        for index, azimuth in enumerate((30, 90, 150, 210, 270, 330))
    )


def ellipsoid_mask(
    camera: Camera, radii: tuple[float, float, float] = (0.6, 0.8, 0.5),
) -> np.ndarray:
    """Render a centered, axis-aligned ellipsoid at pixel centers, without AA.

    Each world-space ray intersects sum((point / radii)**2) = 1.
    A hit requires a real intersection at strictly positive camera depth.
    """
    axes = _radii_array(radii)
    rotation, translation = extrinsics(camera)
    origin = (-rotation.T @ translation) / axes
    v, u = np.indices((camera.height, camera.width), dtype=np.float64)
    focal = camera.height / (2 * np.tan(np.radians(camera.fov_deg) / 2))
    rays = np.stack(((u - (camera.width - 1) / 2) / focal,
                     (v - (camera.height - 1) / 2) / focal,
                     np.ones_like(u)), axis=-1)
    directions = (rays @ rotation) / axes
    a = np.sum(directions**2, axis=-1)
    b = 2 * np.sum(directions * origin, axis=-1)
    c = np.sum(origin**2) - 1
    discriminant = b**2 - 4 * a * c
    far_depth = (-b + np.sqrt(np.maximum(discriminant, 0))) / (2 * a)
    return (discriminant >= 0) & (far_depth > 0)


def ellipsoid_occupancy(
    points_3d: np.ndarray, radii: tuple[float, float, float] = (0.6, 0.8, 0.5),
) -> np.ndarray:
    """Return exact inside/outside labels for an (..., 3) array of world points."""
    axes = _radii_array(radii)
    points = np.asarray(points_3d, dtype=np.float64)
    if points.ndim < 1 or points.shape[-1] != 3 or not np.isfinite(points).all():
        raise ValueError("points_3d must have finite coordinates and shape (..., 3)")
    return np.sum((points / axes)**2, axis=-1) <= 1
