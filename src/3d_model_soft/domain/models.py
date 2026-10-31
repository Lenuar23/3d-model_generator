from dataclasses import dataclass
import math
from numbers import Integral


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
