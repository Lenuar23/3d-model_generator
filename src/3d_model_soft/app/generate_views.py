"""Application scenario for producing foreground views from one photograph."""

from pathlib import Path

from ..domain.interfaces import ViewGenerator


def generate_views(photo: str | Path, seed: int, *, generator: ViewGenerator) -> Path:
    """Delegate to the configured generator and return its completed view bundle.

    The caller supplies the generator; importing this scenario does not load AI
    libraries or models. Validation and generation errors propagate unchanged
    so the UI can report them without treating a failed run as a successful one.
    """
    return generator.generate_views(photo, seed)
