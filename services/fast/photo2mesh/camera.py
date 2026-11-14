"""Orthographic and pinhole cameras aimed at a common scene origin."""
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Camera:
    azimuth: float
    elevation: float
    pixels_per_unit: float
    cx: float
    cy: float
    distance: float | None = None
    rotation: tuple | None = None

    def __post_init__(self) -> None:
        if (not np.isfinite([self.azimuth, self.elevation, self.pixels_per_unit, self.cx, self.cy]).all()
            or abs(self.elevation) >= 89 or not 0 < self.pixels_per_unit <= 10000):
            raise ValueError("Camera values must be finite, with positive scale and elevation between ±89°.")
        if self.distance is not None and (not np.isfinite(self.distance) or self.distance <= 0):
            raise ValueError("Perspective camera distance must be positive and finite.")
        if self.rotation is not None:
            axes = np.asarray(self.rotation)
            if (axes.shape != (3,3) or not np.isfinite(axes).all()
                or not np.allclose(axes @ axes.T, np.eye(3), atol=1e-5)
                or np.linalg.det(axes) <= 0):
                raise ValueError("Camera rotation must be right-handed and orthonormal.")

    @property
    def focal_length(self) -> float | None:
        """Return pinhole focal length in pixels, when applicable."""
        return None if self.distance is None else self.pixels_per_unit * self.distance

    @classmethod
    def perspective(cls, azimuth, elevation, focal_length, cx, cy, distance) -> "Camera":
        """Build a square-pixel camera from focal length and orbit distance."""
        if not np.isfinite(distance) or distance <= 0:
            raise ValueError("Perspective camera distance must be positive and finite.")
        return cls(azimuth, elevation, focal_length / distance, cx, cy, distance)

    @property
    def axes(self) -> np.ndarray:
        """Return horizontal, vertical and toward-camera unit vectors."""
        if self.rotation is not None:
            return np.asarray(self.rotation)
        a, e = np.radians([self.azimuth, self.elevation])
        return np.array([[np.cos(a), 0, -np.sin(a)],
                         [-np.sin(a)*np.sin(e), np.cos(e), -np.cos(a)*np.sin(e)],
                         [np.sin(a)*np.cos(e), np.sin(e), np.cos(a)*np.cos(e)]])

    def project(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return continuous pixel centers and depth increasing toward the camera."""
        u, v, depth = (points @ self.axes.T).T
        scale = self.pixels_per_unit
        if self.distance is not None:
            axial = self.distance - depth
            scale = np.divide(self.focal_length, axial, out=np.full_like(axial, np.nan),
                              where=axial > 1e-6)
        return self.cx + u*scale, self.cy - v*scale, depth

    def facing(self, points: np.ndarray, normals: np.ndarray) -> np.ndarray:
        """Return cosine between each normal and its local view direction."""
        toward = self.axes[2]
        if self.distance is None:
            return normals @ toward
        rays = self.distance * toward - points
        return np.einsum("ij,ij->i", normals, rays) / np.maximum(1e-10, np.linalg.norm(rays, axis=1))

    def pixels(self, points: np.ndarray, shape: tuple[int, int]):
        """Safely map projections to nearest pixels without casting invalid coordinates."""
        x, y, depth = self.project(points)
        h, w = shape
        valid = (x >= -0.5) & (x < w-0.5) & (y >= -0.5) & (y < h-0.5)
        ix = np.floor(np.where(valid, x, 0) + 0.5).astype(int)
        iy = np.floor(np.where(valid, y, 0) + 0.5).astype(int)
        return x, y, depth, ix, iy, valid

