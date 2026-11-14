"""Load and normalize photographs before estimating their foreground."""
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError


@dataclass
class Photo:
    rgb: np.ndarray
    alpha: np.ndarray
    lab: np.ndarray
    original_size: tuple[int, int]
    contrast: float = 1.0


def load_photo(
    source: str | Path | Image.Image, max_side: int = 900, contrast: float = 1.0
) -> Photo:
    """Correct EXIF orientation and prepare an RGB image and a filtered Lab image."""
    if not isinstance(max_side, int) or max_side < 16 or not 0.5 <= contrast <= 2:
        raise ValueError("Use a working size of at least 16 pixels and contrast between 0.5 and 2.")
    if not isinstance(source, (str, Path, Image.Image)):
        raise ValueError("Choose a valid photograph.")
    try:
        if isinstance(source, Image.Image):
            oriented = ImageOps.exif_transpose(source).convert("RGBA")
        else:
            with Image.open(source) as image:
                oriented = ImageOps.exif_transpose(image).convert("RGBA")
    except (OSError, ValueError, UnidentifiedImageError, Image.DecompressionBombError) as error:
        raise ValueError("Could not open this image. Choose a valid photograph.") from error
    original_size = oriented.size
    oriented.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    if min(oriented.size) < 16:
        raise ValueError("The photograph must be at least 16 pixels on each side after resizing.")
    rgba = np.asarray(oriented)
    rgb = np.ascontiguousarray(rgba[:, :, :3])
    filtered = cv2.medianBlur(rgb, 3)
    lab = cv2.cvtColor(filtered.astype(np.float32) / 255, cv2.COLOR_RGB2Lab)
    lab[:, :, 0] = np.clip((lab[:, :, 0] - 50) * contrast + 50, 0, 100)
    return Photo(rgb, rgba[:, :, 3], lab, original_size, contrast)
