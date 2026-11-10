"""Application container assembling domain services and infrastructure adapters."""

from dataclasses import dataclass
from pathlib import Path

from ..domain.interfaces import MeshExporter, ViewGenerator
from ..infrastructure.export.glb_exporter import GlbExporter
from ..infrastructure.storage.views_repository import ViewsRepository
from .generate_views import generate_views
from .reconstruct import reconstruct


@dataclass
class Container:
    """Dependency container configuring pipeline adapters and scenarios."""

    views_repository: ViewsRepository
    mesh_exporter: MeshExporter
    view_generator: ViewGenerator | None = None

    @classmethod
    def default(cls, *, view_generator: ViewGenerator | None = None) -> "Container":
        return cls(
            views_repository=ViewsRepository(),
            mesh_exporter=GlbExporter(),
            view_generator=view_generator,
        )

    def reconstruct(self, views_dir: str | Path, output_path: str | Path | None = None, **kwargs) -> Path:
        return reconstruct(views_dir, output_path, exporter=self.mesh_exporter, **kwargs)

    def generate_views(self, photo: str | Path, seed: int) -> Path:
        if self.view_generator is None:
            raise RuntimeError("ViewGenerator is not configured in this container")
        return generate_views(photo, seed, generator=self.view_generator)
