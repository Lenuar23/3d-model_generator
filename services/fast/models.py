"""Provision the original Photo2Mesh checkpoints on first use."""
from pathlib import Path

from huggingface_hub import snapshot_download

MODELS = Path("/app/models")


def ensure_models():
    clip = MODELS / "clip"
    if not (clip / "model.safetensors").is_file():
        snapshot_download(
            "openai/clip-vit-base-patch32",
            revision="b97b0100e55e367c057773c2a614676470b0d575",
            allow_patterns=["*.json", "merges.txt", "model.safetensors"],
            local_dir=clip,
        )

    wonder = MODELS / "wonder3d"
    if not (wonder / ".download-complete").is_file():
        snapshot_download(
            "flamehaze1115/wonder3d-v1.0",
            revision="d6d2efc033a06a74d3761268de7295c97e6935d2",
            local_dir=wonder,
        )
        (wonder / ".download-complete").touch()
