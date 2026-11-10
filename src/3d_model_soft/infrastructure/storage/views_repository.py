"""Read and write the RGBA view bundle shared with reconstruction."""

from collections.abc import Mapping
import json
from pathlib import Path
import re
from tempfile import TemporaryDirectory

import numpy as np
from PIL import Image, UnidentifiedImageError

from ...domain.models import Camera, View


_FIELDS = ("azimuth", "elevation", "distance", "fov")


def _check_filenames(names):
    folded = set()
    for name in names:
        if (not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*\.png", name)
                or re.fullmatch(r"(?i)(?:con|prn|aux|nul|com[0-9]|lpt[0-9])\.png", name)):
            raise ValueError(f"Invalid view PNG filename: {name!r}")
        if name.casefold() in folded:
            raise ValueError(f"Duplicate view PNG filename: {name}")
        folded.add(name.casefold())


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


class ViewsRepository:
    """Store filename-to-camera JSON and matching RGBA PNGs in a directory."""

    def save(self, views_dir: str | Path, views: Mapping[str, tuple[Camera, np.ndarray]]) -> Path:
        target = Path(views_dir)
        if target.exists() or target.is_symlink():
            raise FileExistsError(f"View bundle already exists: {target}")
        if not views:
            raise ValueError("A view bundle must contain at least one image")
        _check_filenames(views)

        metadata = {}
        for name, (camera, rgba) in views.items():
            if not isinstance(camera, Camera):
                raise ValueError(f"Expected a Camera for {name}")
            if (not isinstance(rgba, np.ndarray) or rgba.dtype != np.uint8
                    or rgba.shape != (camera.height, camera.width, 4)):
                raise ValueError(f"Expected uint8 RGBA matching camera dimensions for {name}")
            metadata[name] = dict(azimuth=float(camera.azimuth_deg), elevation=float(camera.elevation_deg),
                                  distance=float(camera.distance), fov=float(camera.fov_deg))

        target.parent.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(prefix=".views-", dir=target.parent) as temporary:
            staged = Path(temporary) / "bundle"
            staged.mkdir()
            for name in sorted(views):
                Image.fromarray(views[name][1]).save(staged / name, format="PNG")
            (staged / "cameras.json").write_text(
                json.dumps(metadata, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8",
            )
            if target.exists() or target.is_symlink():
                raise FileExistsError(f"View bundle already exists: {target}")
            staged.rename(target)
        return target

    def load(self, views_dir: str | Path, *, alpha_threshold: int = 128) -> tuple[View, ...]:
        """Load views in filename order, using alpha >= threshold as the silhouette."""
        if type(alpha_threshold) is not int or not 1 <= alpha_threshold <= 255:
            raise ValueError("alpha_threshold must be an integer from 1 to 255")
        source = Path(views_dir)
        metadata_path = source / "cameras.json"
        if metadata_path.is_symlink() or not metadata_path.is_file():
            raise ValueError(f"Expected a regular cameras.json file in {source}")
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"), object_pairs_hook=_unique_object)
        except (ValueError, OSError) as exc:
            raise ValueError(f"Invalid cameras.json in {source}: {exc}") from exc
        if not isinstance(metadata, dict) or not metadata:
            raise ValueError("cameras.json must map PNG filenames to camera objects")
        _check_filenames(metadata)
        if {path.name for path in source.iterdir() if path.suffix.lower() == ".png"} != set(metadata):
            raise ValueError("PNG files and cameras.json entries must match")

        views = []
        for name, params in sorted(metadata.items()):
            if (not isinstance(params, dict) or set(params) != set(_FIELDS)
                    or any(type(params[field]) not in (int, float) for field in _FIELDS)):
                raise ValueError(f"Invalid camera fields for {name}")
            path = source / name
            if path.is_symlink() or not path.is_file():
                raise ValueError(f"Expected a regular PNG file: {name}")
            try:
                with Image.open(path) as image:
                    if image.format != "PNG" or image.mode != "RGBA" or image.n_frames != 1:
                        raise ValueError(f"Expected a single RGBA PNG image: {name}")
                    width, height = image.size
                    rgba = np.array(image)
            except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
                raise ValueError(f"Cannot read PNG image: {name}") from exc
            try:
                camera = Camera(*(float(params[field]) for field in _FIELDS), width, height)
            except (ValueError, OverflowError) as exc:
                raise ValueError(f"Invalid camera for {name}: {exc}") from exc
            views.append(View(camera, rgba[..., 3] >= alpha_threshold, rgba[..., :3].copy()))
        return tuple(views)
