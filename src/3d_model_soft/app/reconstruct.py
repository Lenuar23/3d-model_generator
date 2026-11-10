"""Application scenario for reconstructing a 3D GLB model from calibrated views."""

from pathlib import Path

from ..domain.interfaces import MeshExporter
from ..infrastructure.export.glb_exporter import GlbExporter
from ..infrastructure.storage.views_repository import ViewsRepository
from ..services import carving, meshing, texturing


def reconstruct(
    views_dir: str | Path,
    output_path: str | Path | None = None,
    *,
    resolution: int = 64,
    smoothing_iterations: int = 3,
    min_views: int | None = None,
    exporter: MeshExporter | None = None,
) -> Path:
    """Reconstruct a textured 3D mesh from a calibrated view bundle and export to GLB.

    Args:
        views_dir: Path to directory containing view_00..view_05.png and cameras.json.
        output_path: Target path for the .glb file. If None, saves as 'model.glb' inside views_dir.
        resolution: Resolution of the cubic voxel grid (default 64).
        smoothing_iterations: Number of Taubin smoothing iterations (default 3).
        min_views: Number of views required to retain a voxel (None for strict visual hull).
        exporter: MeshExporter implementation to use (defaults to GlbExporter).

    Returns:
        Path to the exported .glb file.
    """
    source = Path(views_dir).resolve(strict=True)
    if output_path is None:
        target = source / "model.glb"
    else:
        target = Path(output_path).resolve()

    # 1. Load calibrated views
    repository = ViewsRepository()
    views = repository.load(source)
    if not views:
        raise ValueError(f"No valid views found in {source}")

    # 2. Carve visual hull on voxel grid
    grid = carving.create_grid(resolution=resolution)
    carved = carving.carve(views, grid, min_views=min_views)

    # 3. Extract mesh via marching cubes, clean small components, and smooth
    mesh = meshing.build_mesh(carved, smoothing_iterations=smoothing_iterations)

    # 4. Transfer vertex colors from visible views
    colored_mesh = texturing.texture_mesh(mesh, views)

    # 5. Export to binary glTF (GLB)
    active_exporter = exporter if exporter is not None else GlbExporter()
    return active_exporter.export(colored_mesh, target)
