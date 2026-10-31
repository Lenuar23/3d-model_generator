"""Perspective projection: world +Y up, image +u right and +v down.

Azimuth 0 places the camera on +Z; +90 places it on +X. Elevation
is measured towards +Y. The camera looks at the origin, with positive
camera-space depth in front of the lens. Integer pixels denote centers.
"""

import numpy as np

from ..domain.models import Camera


def extrinsics(cam: Camera) -> tuple[np.ndarray, np.ndarray]:
    """Return world-to-camera R, t such that camera_point = R @ point + t."""
    elevation = np.radians(cam.elevation_deg)
    azimuth = np.radians(cam.azimuth_deg)

    c = cam.distance * np.array([
        np.cos(elevation) * np.sin(azimuth),
        np.sin(elevation),
        np.cos(elevation) * np.cos(azimuth),
    ])

    # Analytic basis also defines a stable roll at elevation +/-90 degrees.
    forward = -np.array([
        np.cos(elevation) * np.sin(azimuth),
        np.sin(elevation),
        np.cos(elevation) * np.cos(azimuth),
    ])
    right = np.array([np.cos(azimuth), 0.0, -np.sin(azimuth)])
    down = np.cross(forward, right)

    R = np.stack([right, down, forward])
    t = -R @ c
    return R, t


def project(points_3d: np.ndarray, camera: Camera) -> np.ndarray:
    """Project an (N, 3) array to floating-point (N, 2) pixel coordinates.

    The FOV is vertical and pixels are square. Points on or behind the
    camera plane return (nan, nan). Off-image coordinates are left intact;
    callers must check bounds before sampling a silhouette.
    """
    points = np.asarray(points_3d, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points_3d must have shape (N, 3)")
    if not np.isfinite(points).all():
        raise ValueError("points_3d must contain only finite coordinates")

    rotation, translation = extrinsics(camera)
    camera_points = points @ rotation.T + translation
    depth = camera_points[:, 2]
    pixels = np.full((len(points), 2), np.nan, dtype=np.float64)
    visible = depth > 0
    focal_length = camera.height / (2 * np.tan(np.radians(camera.fov_deg) / 2))
    center = np.array([(camera.width - 1) / 2, (camera.height - 1) / 2])
    pixels[visible] = (
        camera_points[visible, :2] / depth[visible, None] * focal_length + center
    )
    return pixels
