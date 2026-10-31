"""Write reproducible synthetic test fixtures using only NumPy and the stdlib."""

import json
from pathlib import Path
import struct
import zlib

import numpy as np

from ..services.synthetic import ellipsoid_mask, ellipsoid_occupancy, zero123_cameras


def _write_rgba_png(path: Path, rgba: np.ndarray) -> None:
    # PNG: 8-bit RGBA, filter 0 per row; keep image encoding out of services.
    def chunk(kind: bytes, payload: bytes) -> bytes:
        return (struct.pack(">I", len(payload)) + kind + payload
                + struct.pack(">I", zlib.crc32(kind + payload)))

    height, width, _ = rgba.shape
    rows = b"".join(b"\x00" + row.tobytes() for row in rgba)
    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(rows)) + chunk(b"IEND", b"")
    )


def generate_dataset(
    output_dir: str | Path, *, radii: tuple[float, float, float] = (0.6, 0.8, 0.5),
    version: str = "v1.2", size: int = 256, resolution: int = 64,
) -> Path:
    """Create six RGBA views, camera metadata and a labeled voxel volume.

    Output must not exist: this prevents accidental replacement of a dataset.
    cameras.json is a fixture-local filename-to-camera mapping; it does not
    establish the production storage format owned by views_repository.
    """
    if isinstance(resolution, bool) or not isinstance(resolution, int) or resolution < 2:
        raise ValueError("resolution must be an integer >= 2")
    # Validate shape/radii before making files.
    ellipsoid_occupancy(np.zeros((1, 3)), radii)
    cameras = zero123_cameras(version, size=size)
    extent = float(max(radii) * 1.25)
    spacing = 2 * extent / resolution
    axis = -extent + (np.arange(resolution) + 0.5) * spacing
    points = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1)
    occupancy = ellipsoid_occupancy(points, radii)

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    views = output / "views"
    views.mkdir()
    metadata = {}
    for index, camera in enumerate(cameras):
        name = f"view_{index:02d}.png"
        mask = ellipsoid_mask(camera, radii)
        rgba = np.zeros((size, size, 4), dtype=np.uint8)
        rgba[mask] = (95, 170, 230, 255)
        _write_rgba_png(views / name, rgba)
        metadata[name] = dict(azimuth=camera.azimuth_deg, elevation=camera.elevation_deg,
                              distance=camera.distance, fov=camera.fov_deg)
    (views / "cameras.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    np.savez_compressed(output / "ground_truth.npz", occupancy=occupancy,
                        x=axis, y=axis, z=axis, radii=np.asarray(radii), spacing=spacing)
    info = dict(object="ellipsoid", radii=list(radii), version=version,
                size=size, resolution=resolution, axis_order="xyz",
                voxel_samples="centers", silhouette_samples="pixel centers, binary, no AA",
                camera_angles_source="https://github.com/SUDO-AI-3D/zero123plus",
                camera_note="Published angles only; distance=4 and vertical FOV=30 are synthetic settings.")
    (output / "dataset.json").write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    return output
