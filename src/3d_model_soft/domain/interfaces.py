"""Contracts used by application scenarios, independent of AI libraries."""

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class BackgroundRemover(Protocol):
    def remove(self, image: np.ndarray) -> np.ndarray:
        """Return a new uint8 RGBA array from a uint8 RGB/RGBA image.

        Input shape is (height, width, 3 or 4), with positive dimensions;
        output shape is (height, width, 4). The input must not be modified.
        Alpha represents foreground coverage: 0 is transparent, 255 is opaque.
        """
        ...
