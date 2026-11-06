"""Convert a Zero123++ v1.2 grid into foreground views with cameras."""

from collections.abc import Callable
from contextlib import suppress
import json
import math
from numbers import Integral, Real
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
from uuid import uuid4

import numpy as np
from PIL import Image

from ...domain.interfaces import BackgroundRemover, ViewGenerator
from ...domain.models import Camera
from ..storage.views_repository import ViewsRepository


TILE_SIZE = 320
GRID_SHAPE = (3 * TILE_SIZE, 2 * TILE_SIZE, 3)
MODEL_ID = "sudo-ai/zero123plus-v1.2"
MODEL_REVISION = "2da07e89919e1a130c9b5add1584c70c7aa065fd"
PIPELINE_ID = "sudo-ai/zero123plus-pipeline"
PIPELINE_REVISION = "983e66d28a3637ddd8e3e2fd8165cdff32230872"
GridBackend = Callable[[Path, int], np.ndarray]


class Zero123ViewGenerator(ViewGenerator):
    """Generate one isolated bundle; masking must retain each tile's framing."""

    def __init__(self, grid_backend: GridBackend, background_remover: BackgroundRemover,
                 output_root: str | Path, *, distance: float = 4.0) -> None:
        if isinstance(distance, bool) or not isinstance(distance, Real) or not math.isfinite(distance) or distance <= 0:
            raise ValueError("distance must be a finite positive number")
        self._grid_backend = grid_backend
        self._background_remover = background_remover
        self._output_root = Path(output_root)
        self._repository = ViewsRepository()
        # Upstream v1.2 poses in the same order as its row-major 2x3 grid.
        self._cameras = tuple(
            Camera(azimuth, (20, -10)[index % 2], float(distance), 30.0, TILE_SIZE, TILE_SIZE)
            for index, azimuth in enumerate((30, 90, 150, 210, 270, 330))
        )

    def generate_views(self, photo: str | Path, seed: int) -> Path:
        source = Path(photo)
        if not source.is_file():
            raise FileNotFoundError(f"Photo does not exist: {source}")
        if isinstance(seed, bool) or not isinstance(seed, Integral) or not 0 <= int(seed) < 2**63:
            raise ValueError("seed must be an integer in [0, 2**63)")
        grid = self._grid_backend(source.resolve(), int(seed))
        if not isinstance(grid, np.ndarray) or grid.dtype != np.uint8 or grid.shape != GRID_SHAPE:
            raise ValueError("Expected a uint8 RGB Zero123++ grid of 640 x 960 pixels")

        views = {}
        for index, camera in enumerate(self._cameras):
            row, col = divmod(index, 2)
            y, x = row * TILE_SIZE, col * TILE_SIZE
            tile = grid[y:y + TILE_SIZE, x:x + TILE_SIZE].copy()
            rgba = self._background_remover.remove(tile)
            views[f"view_{index:02d}.png"] = (camera, rgba)

        run_dir = self._output_root / uuid4().hex
        run_dir.mkdir(parents=True, exist_ok=False)
        try:
            return self._repository.save(run_dir / "views", views)
        except BaseException:
            with suppress(OSError):
                run_dir.rmdir()
            raise


class Zero123SubprocessBackend:
    """Call the task 4 runner in its isolated Python environment."""

    def __init__(self, python: str | Path, runner: str | Path, *, steps: int = 28,
                 vae_tiling: bool = False, timeout: float = 1800) -> None:
        self._python = Path(python).resolve(strict=True)
        self._runner = Path(runner).resolve(strict=True)
        if not self._python.is_file() or not self._runner.is_file():
            raise ValueError("python and runner must be regular files")
        if type(steps) is not int or not 1 <= steps <= 1000:
            raise ValueError("steps must be an integer from 1 to 1000")
        if isinstance(timeout, bool) or not isinstance(timeout, Real) or not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("timeout must be a finite positive number")
        self._steps = steps
        self._vae_tiling = vae_tiling
        self._timeout = timeout

    def __call__(self, photo: Path, seed: int) -> np.ndarray:
        photo = photo.resolve()
        with TemporaryDirectory(prefix="photo3d-zero123-") as temporary:
            output = Path(temporary) / "run"
            command = [str(self._python), "-u", str(self._runner), "--photo", str(photo),
                       "--output-dir", str(output), "--seed", str(seed), "--steps", str(self._steps)]
            if self._vae_tiling:
                command.append("--vae-tiling")
            try:
                subprocess.run(command, check=True, capture_output=True, encoding="utf-8",
                               errors="replace", timeout=self._timeout)
            except subprocess.CalledProcessError as exc:
                diagnostic = (exc.stderr or exc.stdout or str(exc)).strip()[-2000:]
                raise RuntimeError(f"Zero123++ runner failed: {diagnostic}") from exc
            except subprocess.TimeoutExpired as exc:
                raise RuntimeError(f"Zero123++ runner exceeded {self._timeout} seconds") from exc
            except OSError as exc:
                raise RuntimeError(f"Could not start Zero123++ runner: {exc}") from exc

            expected_grid = f"01-{photo.stem}-grid.png"
            try:
                report = json.loads((output / "run.json").read_text(encoding="utf-8"))
                records = report["photos"]
                if (report["status"] != "succeeded" or report["succeeded"] != 1
                        or report["model"] != MODEL_ID or report["model_revision"] != MODEL_REVISION
                        or report["pipeline"] != PIPELINE_ID or report["pipeline_revision"] != PIPELINE_REVISION
                        or report["seed"] != seed or len(records) != 1
                        or records[0]["status"] != "succeeded" or records[0]["grid"] != expected_grid):
                    raise ValueError("Runner did not report the expected successful v1.2 result")
                with Image.open(output / expected_grid) as grid:
                    if grid.format != "PNG" or grid.mode != "RGB" or grid.size != (640, 960) or grid.n_frames != 1:
                        raise ValueError("Runner output is not a single 640 x 960 RGB PNG")
                    return np.array(grid)
            except (OSError, ValueError, TypeError, KeyError, Image.DecompressionBombError) as exc:
                raise RuntimeError(f"Invalid Zero123++ runner output: {exc}") from exc
