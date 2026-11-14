"""Recover a regular radius profile from horizontal foreground runs."""
from dataclasses import dataclass, field

import numpy as np
from scipy.ndimage import gaussian_filter1d, median_filter
from scipy.interpolate import PchipInterpolator


@dataclass
class Profile:
    rows: np.ndarray
    left: np.ndarray
    right: np.ndarray
    centers: np.ndarray
    radii: np.ndarray
    sample_rows: np.ndarray
    sample_centers: np.ndarray
    sample_radii: np.ndarray
    heights: np.ndarray
    warnings: list[str]
    indicators: dict = field(default_factory=dict)


def analyze(
    mask: np.ndarray, samples: int = 96, smoothing: float = 0.018,
    symmetry: bool = False, rounded: bool = False,
) -> Profile:
    """Interpolate row boundaries, reject radius spikes and normalize height to one."""
    if mask.ndim != 2 or samples < 2 or not np.isfinite(smoothing) or smoothing < 0:
        raise ValueError("Expected a 2D mask, at least two samples and nonnegative smoothing.")
    observed = np.flatnonzero(mask.any(axis=1))
    if observed.size < 3:
        raise ValueError("The silhouette needs at least three foreground rows.")
    occupied = mask[observed].astype(bool, copy=False)
    left_observed = occupied.argmax(axis=1)
    right_observed = mask.shape[1] - 1 - occupied[:, ::-1].argmax(axis=1)
    rows = np.arange(observed[0], observed[-1] + 1)
    left = np.interp(rows, observed, left_observed)
    right = np.interp(rows, observed, right_observed)
    centers = (left + right) / 2
    radii = (right - left) / 2
    median = median_filter(radii, size=5, mode="nearest")
    # Preserve deliberate shoulders and tips. Replace only isolated single-row spikes.
    spikes = np.abs(radii[1:-1] - median[1:-1]) > np.maximum(2, median[1:-1] * 0.18)
    corroborated = np.abs(radii[:-2] - radii[2:]) <= np.maximum(1, median[1:-1] * 0.08)
    clean = radii.copy()
    clean[1:-1] = np.where(spikes & corroborated, median[1:-1], radii[1:-1])
    warnings = []
    axis_rows = radii >= max(1, np.median(radii) * 0.25)
    axis_centers = median_filter(centers, size=5, mode="nearest")
    drift = float(np.ptp(np.percentile(axis_centers[axis_rows], [5, 95]))) if axis_rows.any() else 0
    if drift > max(4, radii.max() * 0.35):
        warnings.append("Strong centerline drift suggests tilt or asymmetry. The model assumes an upright axis.")
    gaps = right_observed - left_observed + 1 - occupied.sum(axis=1)
    if observed.size < rows.size or np.any(gaps):
        warnings.append("The silhouette contains gaps. The profile uses the outer boundaries.")
    large_gap = gaps > np.maximum(4, 0.18 * (right_observed - left_observed + 1))
    if np.count_nonzero(large_gap) > max(2, 0.04 * len(rows)):
        warnings.append("Repeated large gaps may indicate a handle, opening, or missing foreground. "
                        "The rotational mesh cannot represent these features.")
    end_span = max(2, round(len(rows) * 0.06))
    bottom_flare = float(max(0, clean[-end_span:].max() - clean[-2*end_span:-end_span].max())) if len(rows) >= 2 * end_span else 0
    if bottom_flare > max(4, clean.max() * 0.14):
        warnings.append("The silhouette widens near the base. Check for a shadow or reflected foreground.")
    if len(rows) / max(1, 2 * radii.max()) < 0.6:
        warnings.append("The silhouette is very wide for the upright model. Check orientation and symmetry.")
    if symmetry:
        axis = np.median(axis_centers[axis_rows]) if axis_rows.any() else np.median(centers)
        clean = np.maximum(0, clean - np.abs(centers - axis))
    if smoothing:
        original = clean
        blurred = gaussian_filter1d(original, max(0.15, smoothing * len(rows)), mode="nearest")
        tolerance = max(1, 0.04 * original.max())
        weight = 1 / (1 + ((blurred - original) / tolerance) ** 2)
        clean = original + weight * (blurred - original)
        clean[[0, -1]] = original[[0, -1]]
    # Bottom-to-top ordering matches positive model Y.
    sample_rows = np.linspace(rows[-1], rows[0], samples)
    height = float(rows[-1] - rows[0])
    sampled = np.interp(sample_rows, rows, clean)
    if rounded:
        # Smooth poles have approximately linear squared radius near their tips.
        squared = PchipInterpolator(rows, clean**2)(sample_rows)
        for start in (True, False):
            index = 0 if start else -1
            if clean[index] >= 0.2 * clean.max():
                continue
            distance = np.abs(sample_rows - rows[index]) / height
            weight = np.clip(1 - distance / 0.12, 0, 1)
            sampled = sampled * (1 - weight) + np.sqrt(np.maximum(0, squared)) * weight
        if max(clean[0], clean[-1]) >= 0.2 * clean.max():
            warnings.append("Rounded mode preserved a broad measured end. Inspect the contour before changing the cap.")
    return Profile(
        rows, left, right, centers, radii, sample_rows,
        np.interp(sample_rows, rows, centers),
        np.maximum(0.5, sampled) / height,
        np.linspace(-0.5, 0.5, samples), warnings, {
            "centerline_drift_pixels": round(drift, 2),
            "missing_rows": int(rows.size - observed.size),
            "rows_with_large_gaps": int(np.count_nonzero(large_gap)),
            "bottom_flare_pixels": round(bottom_flare, 2),
        },
    )
