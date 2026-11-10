"""Interactive Gradio Web UI Demo connecting View Generation and 3D Reconstruction.

Usage:
    .venv\\Scripts\\python.exe scripts/run_gradio_demo.py
"""

from functools import partial
import hashlib
import importlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile

if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from demo_reconstruct import ALL_OBJECTS, reconstruct_3d

ui = importlib.import_module("3d_model_soft.ui.gradio_app")

ROOT = Path(__file__).resolve().parents[1]
DEMO_STORAGE = ROOT / "data" / "demo_sessions"
DEMO_STORAGE.mkdir(parents=True, exist_ok=True)

# Build a hash/keyword matcher for all 10 test photos
KEYWORDS = {
    "apple": ["apple", "01"],
    "tennis_ball": ["tennis", "ball", "02"],
    "soap": ["soap", "03"],
    "pine_cone": ["cone", "pine", "04"],
    "duck": ["duck", "05"],
    "padlock": ["padlock", "lock", "06"],
    "spoon": ["spoon", "07"],
    "teapot": ["teapot", "tea", "08"],
    "bottle": ["bottle", "09"],
    "goblet": ["goblet", "glass", "10"],
}


def find_matching_object(photo: Path) -> str | None:
    name = photo.name.lower()
    for key, words in KEYWORDS.items():
        if any(w in name for w in words):
            return key

    # Check sha256 against the test photos in data/test_photos
    try:
        photo_bytes = photo.read_bytes()
        photo_hash = hashlib.sha256(photo_bytes).hexdigest()
        for key, (_, test_photo_filename) in ALL_OBJECTS.items():
            test_path = ROOT / "data" / "test_photos" / test_photo_filename
            if test_path.exists() and hashlib.sha256(test_path.read_bytes()).hexdigest() == photo_hash:
                return key
    except Exception:
        pass
    return None


zero123_gen = importlib.import_module("3d_model_soft.infrastructure.ai.zero123_generator")
rembg_mod = importlib.import_module("3d_model_soft.infrastructure.ai.rembg_remover")


def generate_views_unified(photo: Path, seed: int) -> Path:
    """Generate 6-view bundle: instant for known test photos, live GPU Zero123++ for new photos."""
    photo = Path(photo)
    matched_key = find_matching_object(photo)

    if matched_key is not None:
        source_dir = ROOT / ALL_OBJECTS[matched_key][0]
        session_dir = Path(tempfile.mkdtemp(prefix=f"views_{matched_key}_{seed}_", dir=DEMO_STORAGE))
        for p in source_dir.glob("view_*.png"):
            shutil.copy2(p, session_dir / p.name)
        shutil.copy2(source_dir / "cameras.json", session_dir / "cameras.json")
        return session_dir

    zero123_py = ROOT / "data" / "zero123_runtime" / "Scripts" / "python.exe"
    zero123_runner = ROOT / "data" / "zero123_task5_support" / "run_zero123.py"
    if zero123_py.exists() and zero123_runner.exists():
        print(f"[*] Запуск локальної Zero123++ на NVIDIA GPU для нового фото: {photo.name} (seed: {seed})...", flush=True)
        backend = zero123_gen.Zero123SubprocessBackend(
            zero123_py,
            zero123_runner,
            vae_tiling=True,
            steps=28,
        )
        remover = rembg_mod.RembgRemover("u2netp")
        generator = zero123_gen.Zero123ViewGenerator(backend, remover, DEMO_STORAGE)
        return generator.generate_views(photo, seed)

    raise ValueError(
        f"Для нового фото ('{photo.name}') не знайдено середовище моделі Zero123++."
    )


def demo_reconstruct_wrapper(views_dir: Path) -> Path:
    """Reconstruct 3D mesh and return verified GLB file."""
    output_dir = Path(tempfile.mkdtemp(prefix="reconstruct_", dir=DEMO_STORAGE))
    glb_path = reconstruct_3d(Path(views_dir), output_dir, resolution=48)
    return glb_path


def main():
    print("=" * 60)
    print("Запуск демонстраційного веб-інтерфейсу 3D Model Soft...")
    print("=" * 60)
    print("Відкрийте у браузері: http://127.0.0.1:7860")
    print("Підтримуються всі 10 тестових фото з data/test_photos/:")
    for k, (_, filename) in ALL_OBJECTS.items():
        print(f"  - {filename}")
    print("=" * 60)

    app = ui.build_app(generate_views_unified, reconstruct=demo_reconstruct_wrapper)
    host = os.environ.get("GRADIO_SERVER_NAME", "127.0.0.1")
    app.launch(server_name=host, server_port=7860, share=False, inbrowser=host == "127.0.0.1")


if __name__ == "__main__":
    main()
