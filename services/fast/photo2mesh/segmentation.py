"""Classical border-color segmentation with optional correction strokes."""
from dataclasses import dataclass, field

import cv2
import numpy as np
from PIL import ImageColor
from scipy.ndimage import binary_fill_holes

from .image_processing import Photo


@dataclass
class Segmentation:
    mask: np.ndarray
    contour: np.ndarray
    bbox: tuple[int, int, int, int]
    threshold: float | None
    warnings: list[str]
    indicators: dict = field(default_factory=dict)


def background_distance(lab: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit a robust planar lightness trend to the perimeter and keep constant chroma."""
    height, width = lab.shape[:2]
    ys, xs = np.nonzero(np.pad(np.zeros((height - 2, width - 2), dtype=bool), 1,
                               constant_values=True))
    border = lab[ys, xs]
    reference = np.median(border, axis=0)
    x = np.linspace(-1, 1, width, dtype=np.float32)
    y = np.linspace(-1, 1, height, dtype=np.float32)
    design = np.column_stack((np.ones(len(xs)), x[xs], y[ys]))
    keep = np.ones(len(xs), dtype=bool)
    for _ in range(3):
        coefficients = np.linalg.lstsq(design[keep], border[keep, 0], rcond=None)[0]
        residual = np.abs(border[:, 0] - design @ coefficients)
        keep = residual <= max(2, 2.5 * np.median(residual))
    lightness = np.full((height, width), reference[0], dtype=np.float32)
    # Prefer a constant when the perimeter does not support a clear gradient.
    if np.median(residual) + 1 < 0.75 * np.median(np.abs(border[:, 0] - reference[0])):
        lightness[:] = np.clip(coefficients[0] + coefficients[1] * x
                              + coefficients[2] * y[:, None], 0, 100)
    delta = lab - reference
    delta[..., 0] = lab[..., 0] - lightness
    distance = np.sqrt(np.einsum("ijk,ijk->ij", delta, delta))
    return distance, distance[ys, xs], delta


def shadow_candidates(delta: np.ndarray, foreground: np.ndarray, strength: float) -> np.ndarray:
    """Find weak shadow-colored foreground beyond the lower edge of a strong core."""
    chroma = np.hypot(delta[..., 1], delta[..., 2])
    darkness = -delta[..., 0]
    # Deep dark pixels remain eligible as core, even on a neutral background.
    weak = foreground & (chroma <= 4 + 8 * strength) & (darkness > 0)
    weak &= darkness < 25 + 20 * strength
    core = (foreground & ~weak).astype(np.uint8)
    core = cv2.morphologyEx(core, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(core, connectivity=8)
    if count < 2:
        return np.zeros_like(foreground)
    label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    _, top, _, height, area = stats[label]
    if area < 0.25 * foreground.sum():
        return np.zeros_like(foreground)
    # A core-supported row protects dark rims and bases inside the object span.
    row = np.arange(len(core))[:, None]
    cutoff = top + height - 1 - round(height * 0.05 * strength)
    # Above the cutoff, reject only lateral spill outside the supported row.
    lower_start = top + height - 1 - round(height * 0.22 * strength)
    lower_core = labels[max(0, lower_start):cutoff] == label
    candidate = weak & (row >= cutoff)
    if lower_core.size:
        left = lower_core.argmax(axis=1)
        right = core.shape[1] - 1 - lower_core[:, ::-1].argmax(axis=1)
        cols = np.arange(core.shape[1])
        outside = (cols < left[:, None] - 2) | (cols > right[:, None] + 2)
        candidate[lower_start:cutoff] = (
            weak[lower_start:cutoff] & outside & lower_core.any(axis=1)[:, None]
        )
    return candidate


def segment(
    photo: Photo,
    threshold: float = 0,
    background: str | None = None,
    add: np.ndarray | None = None,
    remove: np.ndarray | None = None,
    shadow_strength: float = 0.6,
    mask_smoothing: float = 1,
) -> Segmentation:
    """Select the largest meaningful connected region after cleanup and edits."""
    height, width = photo.alpha.shape
    lab = photo.lab
    warnings = []
    if not 0 <= shadow_strength <= 1 or not 0 <= mask_smoothing <= 2:
        raise ValueError("Use shadow strength between 0 and 1 and mask smoothing between 0 and 2.")
    removed_shadow = 0
    border_outliers = None
    for edit in (add, remove):
        if edit is not None and (edit.shape != photo.alpha.shape or edit.dtype != bool):
            raise ValueError("Correction masks must be boolean arrays matching the image.")
    if not np.any(photo.alpha >= 128):
        raise ValueError("The image is fully transparent or has no sufficiently opaque pixels.")
    used_threshold = None
    if background is None and (photo.alpha < 32).mean() >= 0.01:
        raw = photo.alpha >= 128
    else:
        if background:
            color = np.array([[ImageColor.getcolor(background, "RGB")]], dtype=np.float32)
            reference = cv2.cvtColor(color / 255, cv2.COLOR_RGB2Lab)[0, 0]
            reference[0] = np.clip((reference[0] - 50) * photo.contrast + 50, 0, 100)
            delta = lab - reference
            distance = np.linalg.norm(delta, axis=2)
            border_distance = np.concatenate((distance[0], distance[-1],
                                              distance[1:-1, 0], distance[1:-1, -1]))
        else:
            distance, border_distance, delta = background_distance(lab)
        used_threshold = threshold or float(np.clip(np.median(border_distance) * 3 + 8, 10, 45))
        raw = (distance > used_threshold) & (photo.alpha >= 128)
        border_outliers = float(np.mean(border_distance > used_threshold))
        if border_outliers > 0.15:
            warnings.append("Low segmentation confidence. The background varies around the border "
                            "or the object is clipped. Inspect the mask.")
        if shadow_strength:
            shadow = shadow_candidates(delta, raw, shadow_strength)
            removed_shadow = int(shadow.sum())
            raw &= ~shadow
            if removed_shadow:
                warnings.append("Possible cast shadow removed near the base. Inspect the mask and reduce "
                                "shadow removal if part of the object is missing.")

    radius = round(max(1, round(min(height, width) * 0.004)) * mask_smoothing)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1,) * 2)
    clean = raw.astype(np.uint8)
    if used_threshold is not None:
        clean = cv2.morphologyEx(clean, cv2.MORPH_OPEN, kernel)
    clean = cv2.morphologyEx(clean, cv2.MORPH_CLOSE, kernel)
    if add is not None:
        clean[add] = 1
    if remove is not None:
        clean[remove] = 0
    count, labels, stats, _ = cv2.connectedComponentsWithStats(clean, connectivity=8)
    if count < 2:
        raise ValueError("No foreground found. Try a lower color threshold or paint foreground strokes.")
    label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    mask = labels == label
    if add is not None:
        # A deliberate stroke may bridge a missed row or mark a disconnected tip.
        chosen = np.unique(labels[add & (labels > 0)])
        mask |= np.isin(labels, chosen)
    filled = binary_fill_holes(mask)
    holes_count, holes, hole_stats, _ = cv2.connectedComponentsWithStats(
        (filled & ~mask).astype(np.uint8), connectivity=8
    )
    small = hole_stats[:, cv2.CC_STAT_AREA] <= max(12, mask.sum() * 0.005)
    if holes_count > 1:
        small[0] = False
        mask |= small[holes]
    if remove is not None:
        mask[remove] = False
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(mask.astype(np.uint8))
    coverage = mask.mean()
    if min(w, h) < 5 or coverage < 0.001:
        raise ValueError("The foreground is too small. Use a closer photo or adjust the mask.")
    if coverage > 0.8:
        warnings.append("The foreground covers over 80% of the image. Check the mask and background.")
    if x == 0 or y == 0 or x + w == width or y + h == height:
        warnings.append("The silhouette touches the image edge and may be clipped.")
    if stats[1:, cv2.CC_STAT_AREA].sum() > 1.2 * stats[label, cv2.CC_STAT_AREA]:
        warnings.append("Other sizable regions were discarded. Check that the intended object was selected.")
    component_count, _ = cv2.connectedComponents(mask.astype(np.uint8))
    if component_count > 2:
        warnings.append("The mask has disconnected foreground. Check whether the added regions belong to the object.")
    return Segmentation(mask, contour, (x, y, w, h), used_threshold, warnings, {
        "shadow_candidate_pixels_removed": removed_shadow,
        "border_outlier_fraction": border_outliers,
        "mask_components": component_count - 1,
        "mask_area_pixels": int(mask.sum()),
        "outer_perimeter_pixels": round(cv2.arcLength(contour, True), 2),
    })
