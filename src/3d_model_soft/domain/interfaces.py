"""Boundaries implemented by external format/model adapters."""

from pathlib import Path
from typing import Protocol

from .models import Mesh


class MeshExporter(Protocol):
    """Write a mesh to a portable file and return its path."""

    def export(self, mesh: Mesh, output_path: str | Path) -> Path: ...
