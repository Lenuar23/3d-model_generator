"""Lazy CPU rembg adapter; importing this module never downloads a model."""

import importlib

import numpy as np

from ...domain.interfaces import BackgroundRemover


class RembgRemover(BackgroundRemover):
    """Remove backgrounds with one reusable, lazily created local session.

    Model weights are downloaded by rembg on the first valid remove() call.
    Instances are intended for sequential use, not concurrent initialization.
    """

    def __init__(self, model_name: str = "u2net") -> None:
        if not isinstance(model_name, str) or not model_name.strip():
            raise ValueError("model_name must be a non-empty string")
        self._model_name = model_name
        self._backend = None
        self._session = None

    def remove(self, image: np.ndarray) -> np.ndarray:
        if (not isinstance(image, np.ndarray) or image.dtype != np.uint8
                or image.ndim != 3 or image.shape[2] not in (3, 4)
                or image.shape[0] == 0 or image.shape[1] == 0):
            raise ValueError("image must be a non-empty uint8 RGB or RGBA array (H, W, 3 or 4)")

        if self._backend is None:
            try:
                self._backend = importlib.import_module("rembg")
            except (ImportError, SystemExit) as exc:
                # rembg exits during import when its ONNX backend is missing.
                raise RuntimeError(
                    'Background removal requires the CPU extra: uv sync --extra background-removal '
                    '(or python -m pip install -e ".[background-removal]").'
                ) from exc
        if self._session is None:
            try:
                self._session = self._backend.new_session(
                    self._model_name, providers=["CPUExecutionProvider"],
                )
            except Exception as exc:
                raise RuntimeError(
                    f"Could not initialize rembg model '{self._model_name}'. "
                    "Check the model name, network access and writable rembg model cache."
                ) from exc

        try:
            # rembg/Pillow may mutate their argument. Keep the caller's data intact.
            result = self._backend.remove(image.copy(), session=self._session)
        except Exception as exc:
            raise RuntimeError("rembg background removal failed") from exc
        if (not isinstance(result, np.ndarray) or result.dtype != np.uint8
                or result.shape != (*image.shape[:2], 4)):
            raise RuntimeError("rembg must return a same-size uint8 RGBA array (H, W, 4)")
        return result.copy()
