from dataclasses import dataclass
import math
from numbers import Integral

import numpy as np


@dataclass(frozen=True)
class Camera:
    """Origin-facing camera with a vertical field of view in degrees."""

    azimuth_deg: float
    elevation_deg: float
    distance: float
    fov_deg: float
    width: int
    height: int

    def __post_init__(self) -> None:
        for name in ("azimuth_deg", "elevation_deg", "distance", "fov_deg"):
            if not math.isfinite(getattr(self, name)):
                raise ValueError(f"{name} must be finite")
        if self.distance <= 0:
            raise ValueError("distance must be positive")
        if not -90 <= self.elevation_deg <= 90:
            raise ValueError("elevation_deg must be between -90 and 90")
        if not 0 < self.fov_deg < 180:
            raise ValueError("fov_deg must be between 0 and 180 (exclusive)")
        for name in ("width", "height"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")


@dataclass(frozen=True)
class View:
    """Calibrated silhouette; mask is boolean with shape (height, width)."""

    camera: Camera
    mask: np.ndarray

    def __post_init__(self) -> None:
        if not isinstance(self.camera, Camera):
            raise ValueError("camera must be a Camera")
        if (not isinstance(self.mask, np.ndarray) or self.mask.dtype != np.bool_
                or self.mask.shape != (self.camera.height, self.camera.width)):
            raise ValueError("mask must be a boolean array matching camera height/width")


@dataclass(frozen=True)
class VoxelGrid:
    """XYZ occupancy; origin is the minimum corner, spacing is per-axis size.

    Sample (i, j, k) represents origin + (index + 0.5) * spacing.
    """

    occupancy: np.ndarray
    origin: tuple[float, float, float]
    spacing: tuple[float, float, float]

    def __post_init__(self) -> None:
        if (not isinstance(self.occupancy, np.ndarray) or self.occupancy.dtype != np.bool_
                or self.occupancy.ndim != 3 or min(self.occupancy.shape) < 1):
            raise ValueError("occupancy must be a nonempty 3D boolean array")
        for name in ("origin", "spacing"):
            values = np.asarray(getattr(self, name), dtype=float)
            if values.shape != (3,) or not np.isfinite(values).all():
                raise ValueError(f"{name} must contain three finite values")
            if name == "spacing" and (values <= 0).any():
                raise ValueError("spacing must be positive")


@dataclass(frozen=True)
class Mesh:
    """World-space vertices (N, 3) and outward-wound triangle indices (M, 3)."""

    vertices: np.ndarray
    faces: np.ndarray

    def __post_init__(self) -> None:
        if (not isinstance(self.vertices, np.ndarray) or self.vertices.ndim != 2
                or self.vertices.shape[1] != 3 or self.vertices.dtype.kind != "f"
                or not np.isfinite(self.vertices).all()):
            raise ValueError("vertices must be a finite floating-point (N, 3) array")
        if (not isinstance(self.faces, np.ndarray) or self.faces.ndim != 2
                or self.faces.shape[1] != 3 or self.faces.dtype.kind not in "iu"):
            raise ValueError("faces must be an integer (M, 3) array")
        if self.faces.size and (self.faces.min() < 0 or self.faces.max() >= len(self.vertices)):
            raise ValueError("face indices must reference existing vertices")
