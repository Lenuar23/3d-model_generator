"""Session workflow for previewing and confirming generated views."""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from secrets import randbits
import struct
from threading import Lock


@dataclass(frozen=True)
class GeneratedViews:
    directory: Path
    description: str


class GenerationUnavailable(RuntimeError):
    """A generation error whose message can be shown to the user."""


GenerateViews = Callable[[Path, int], Path | GeneratedViews]
Reconstruct = Callable[[Path], Path]


@dataclass
class ViewSession:
    photo: Path | None = None
    seed: int | None = None
    views_dir: Path | None = None
    revision: int = 0
    busy: bool = False


@dataclass(frozen=True)
class GenerationRequest:
    photo: Path
    seed: int
    revision: int
    regenerate: bool = False


@dataclass(frozen=True)
class ReconstructionRequest:
    views_dir: Path
    revision: int


class ViewWorkflow:
    """Coordinate generation callbacks without coupling the app to Gradio or AI."""

    def __init__(self, generate: GenerateViews, reconstruct: Reconstruct | None = None, *,
                 regenerate: GenerateViews | None = None) -> None:
        self._generate = generate
        self._regenerate = regenerate or generate
        self._reconstruct = reconstruct
        self._lock = Lock()

    @property
    def can_reconstruct(self) -> bool:
        return self._reconstruct is not None

    def set_photo(self, session: ViewSession, photo: str | Path | None) -> None:
        with self._lock:
            session.revision += 1
            session.photo = Path(photo).resolve() if photo is not None else None
            session.seed = None
            session.views_dir = None
            session.busy = False

    def begin_generation(self, session: ViewSession, photo: str | Path | None, *,
                         regenerate: bool = False) -> GenerationRequest | None:
        source = Path(photo).resolve() if photo is not None else None
        with self._lock:
            if source is None or source != session.photo or session.busy:
                return None
            if regenerate != (session.views_dir is not None):
                return None
            seed = randbits(63)
            if seed == session.seed:
                seed = (seed + 1) % (2**63)
            session.seed = seed
            session.revision += 1
            session.busy = True
            return GenerationRequest(source, seed, session.revision, regenerate)

    def generate(self, request: GenerationRequest) -> tuple[Path, tuple[Path, ...]]:
        result, previews = self.generate_with_description(request)
        return result.directory, previews

    def generate_with_description(self, request: GenerationRequest) -> tuple[GeneratedViews, tuple[Path, ...]]:
        callback = self._regenerate if request.regenerate else self._generate
        result = callback(request.photo, request.seed)
        if not isinstance(result, GeneratedViews):
            result = GeneratedViews(result, f"Згенеровано 6 ракурсів. Seed: {request.seed}.")
        directory = Path(result.directory).resolve(strict=True)
        manifest = directory / "cameras.json"
        previews = tuple(directory / f"view_{index:02d}.png" for index in range(6))
        if any(path.is_symlink() or not path.is_file() for path in (manifest, *previews)):
            raise ValueError("Expected six view PNG files and cameras.json")
        return GeneratedViews(directory, result.description), previews

    def finish_generation(self, session: ViewSession, request: GenerationRequest,
                          views_dir: Path | None) -> bool:
        with self._lock:
            if request.revision != session.revision:
                return False
            if views_dir is not None:
                session.views_dir = views_dir
            session.busy = False
            return True

    def begin_reconstruction(self, session: ViewSession,
                             photo: str | Path | None) -> ReconstructionRequest | None:
        source = Path(photo).resolve() if photo is not None else None
        with self._lock:
            if (not self.can_reconstruct or source is None or source != session.photo
                    or session.views_dir is None or session.busy):
                return None
            session.revision += 1
            session.busy = True
            return ReconstructionRequest(session.views_dir, session.revision)

    def reconstruct(self, request: ReconstructionRequest) -> Path:
        if self._reconstruct is None:
            raise RuntimeError("Reconstruction is unavailable")
        path = Path(self._reconstruct(request.views_dir))
        if path.suffix.lower() != ".glb" or path.is_symlink() or not path.is_file():
            raise ValueError("Reconstruction must return an existing GLB file")
        with path.open("rb") as stream:
            header = stream.read(12)
            if len(header) != 12:
                raise ValueError("Reconstruction returned an incomplete GLB header")
            magic, version, declared_size = struct.unpack("<4sII", header)
            if magic != b"glTF" or version != 2 or declared_size != path.stat().st_size:
                raise ValueError("Reconstruction returned an invalid GLB header")
        return path.resolve()

    def finish_reconstruction(self, session: ViewSession, request: ReconstructionRequest) -> bool:
        with self._lock:
            if request.revision != session.revision:
                return False
            session.busy = False
            return True
