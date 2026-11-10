"""Contracts used by application scenarios, independent of AI libraries."""

from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np

from .models import Mesh


@runtime_checkable
class BackgroundRemover(Protocol):
    def remove(self, image: np.ndarray) -> np.ndarray:
        """Return a new uint8 RGBA array from a uint8 RGB/RGBA image.

        Input shape is (height, width, 3 or 4), with positive dimensions;
        output shape is (height, width, 4). The input must not be modified.
        Alpha represents foreground coverage: 0 is transparent, 255 is opaque.
        """
        ...


@runtime_checkable
class ViewGenerator(Protocol):
    def generate_views(self, photo: str | Path, seed: int) -> Path:
        """Generate foreground views with camera metadata and return their directory."""
        ...


@runtime_checkable
class MeshExporter(Protocol):
    """Write a mesh to a portable file and return its path."""

    def export(self, mesh: Mesh, output_path: str | Path) -> Path: ...
