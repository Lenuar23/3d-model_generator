"""Robust blending for inferred colors that disagree after projection."""
import numpy as np


def blend_inferred(samples: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Downweight disagreement smoothly around the weighted color center.

    Arrays are view × surface point × RGB and view × surface point.
    A continuous Cauchy kernel avoids abrupt surface patches when views swap rank.
    """
    if len(samples) == 1:
        return samples[0]
    anchor = np.einsum("vp,vpc->pc", weights, samples) / np.maximum(weights.sum(axis=0)[:,None], 1e-12)
    deviation = np.sum((samples - anchor)**2, axis=-1)
    robust = weights / (1 + deviation / 48**2)
    total = robust.sum(axis=0)
    return np.einsum("vp,vpc->pc", robust, samples) / np.maximum(total[:, None], 1e-12)

