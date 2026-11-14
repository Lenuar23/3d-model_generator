"""Run local CLIP recognition in its isolated interpreter.

View generation is handled by ``wonder3d_adapter``. This module only provides the
shared project root and the CPU recognition worker used by the AI-Assisted workflow.
"""
import json
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
WORKER = Path(__file__).with_name("model_worker.py")
CLIP_MODEL = "openai/clip-vit-base-patch32@b97b0100e55e367c057773c2a614676470b0d575"


def worker(operation, images, output):
    """Call the isolated recognition interpreter and surface errors from local inference."""
    if operation != "recognize":
        raise ValueError(f"Unsupported local operation: {operation}.")
    python = ROOT / ".venv-recognition" / "Scripts/python.exe"
    model = ROOT / "models" / "clip"
    if not python.is_file() or not model.is_dir():
        raise ValueError(f"Install the local {operation} environment and checkpoint as described in the AI guide.")
    command = [str(python), str(WORKER), operation, *map(str, images), "--model", str(model),
               "--output", str(output)]
    output.mkdir(parents=True, exist_ok=True)
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=900,
                                env={**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                                     "PYTHONIOENCODING": "utf-8"}, encoding="utf-8",
                                errors="replace")
    except subprocess.TimeoutExpired as error:
        raise ValueError(f"Local {operation} exceeded 15 minutes.") from error
    (output / "worker.log").write_text(result.stdout + result.stderr, encoding="utf-8")
    if result.returncode:
        lines = [line for line in result.stderr.splitlines() if line.strip()]
        detail = lines[-1] if lines else "The worker exited before returning a result."
        raise ValueError(f"Local {operation} failed (exit {result.returncode}). {detail} Log: {output / 'worker.log'}")
    return json.loads((output / "result.json").read_text(encoding="utf-8"))
