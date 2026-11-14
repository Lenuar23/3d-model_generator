"""Local-only adapters for recognition and externally generated image views."""
from dataclasses import dataclass
from functools import lru_cache
import json
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from PIL import Image

if TYPE_CHECKING:
    from .camera import Camera

CATEGORIES = ("ball", "apple", "bottle", "vase", "shoe", "cup", "box",
              "padlock", "thin object", "other or unknown")


class Recognizer(Protocol):
    def classify(self, image: Image.Image) -> tuple[str, float]: ...


@dataclass(frozen=True)
class GeneratedView:
    path: Path
    camera: "Camera"
    model: str


class ViewGenerator(Protocol):
    def generate(self, image: Image.Image, destination: Path) -> list[GeneratedView]: ...


@lru_cache(maxsize=1)
def _load_clip(model_path: str):
    try:
        from transformers import CLIPModel, CLIPProcessor
    except ImportError as error:
        raise ValueError("Recognition requires the optional AI environment with PyTorch and Transformers.") from error
    try:
        processor = CLIPProcessor.from_pretrained(model_path, local_files_only=True)
        model = CLIPModel.from_pretrained(model_path, local_files_only=True, use_safetensors=True)
    except (OSError, ValueError) as error:
        raise ValueError("Could not load the local CLIP snapshot. Check the configured model directory.") from error
    return model.eval(), processor


@dataclass(frozen=True)
class LocalClip:
    """Run CLIP zero-shot recognition on CPU with an explicitly provisioned snapshot."""
    directory: Path

    def classify(self, image: Image.Image) -> tuple[str, float]:
        if not self.directory.is_dir():
            raise ValueError("Set PHOTO2MESH_CLIP_DIR to an existing local CLIP snapshot.")
        model, processor = _load_clip(str(self.directory.resolve()))
        import torch

        prompts = [f"a photograph of a single {category}" for category in CATEGORIES]
        prompts[-1] = "a photograph of an unidentified object"
        with torch.inference_mode():
            scores = model(**processor(text=prompts, images=image.convert("RGB"),
                                       return_tensors="pt", padding=True)).logits_per_image[0]
        values, indices = scores.topk(2)
        margin = float(values[0] - values[1])
        return (CATEGORIES[int(indices[0])] if margin >= 2 else "other or unknown", margin)


def read_generated_manifest(path: Path) -> list[GeneratedView]:
    """Import generated images with explicit, user-reviewed orthographic metadata."""
    from .camera import Camera

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, list) or not 2 <= len(payload) <= 7:
            raise ValueError("Supply a list of 2–7 generated view records.")
        entries = []
        for record in payload:
            if record["projection"] != "orthographic" or record["pose_status"] != "reviewed":
                raise ValueError("Only reviewed orthographic poses can enter this engine.")
            model = record["model"]
            if not isinstance(model, str) or not model.strip():
                raise ValueError("Record the generator name and version.")
            root = path.parent.resolve()
            image = (root / record["image"]).resolve()
            if not image.is_relative_to(root) or not image.is_file():
                raise ValueError("Generated image must exist within the manifest directory.")
            camera = Camera(*(float(record[key]) for key in
                              ("azimuth", "elevation", "pixels_per_unit", "cx", "cy")))
            entries.append(GeneratedView(image, camera, model))
        return entries
    except (KeyError, TypeError, OSError, ValueError) as error:
        raise ValueError(f"Invalid generated-view manifest. {error}") from error

