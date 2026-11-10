"""Interactive Gradio Web UI Demo connecting View Generation and 3D Reconstruction.

Usage:
    .venv\\Scripts\\python.exe scripts/run_gradio_demo.py
"""

import importlib
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
workflow = importlib.import_module("3d_model_soft.app.view_workflow")

ROOT = Path(__file__).resolve().parents[1]
DEMO_STORAGE = ROOT / "data" / "demo_sessions"
DEMO_STORAGE.mkdir(parents=True, exist_ok=True)

def find_matching_object(photo: Path) -> str | None:
    contents = photo.read_bytes()
    for key, (_, filename) in ALL_OBJECTS.items():
        reference = ROOT / "data" / "test_photos" / filename
        if reference.is_file() and reference.read_bytes() == contents:
            return key
    return None


zero123_gen = importlib.import_module("3d_model_soft.infrastructure.ai.zero123_generator")
rembg_mod = importlib.import_module("3d_model_soft.infrastructure.ai.rembg_remover")


def generate_views_unified(photo: Path, seed: int) -> workflow.GeneratedViews:
    matched_key = find_matching_object(photo)
    if matched_key is not None:
        source_dir = ROOT / ALL_OBJECTS[matched_key][0]
        files = [source_dir / f"view_{index:02d}.png" for index in range(6)]
        files.append(source_dir / "cameras.json")
        if all(path.is_file() for path in files):
            session_dir = Path(tempfile.mkdtemp(prefix=f"views_{matched_key}_", dir=DEMO_STORAGE))
            for path in files:
                shutil.copy2(path, session_dir / path.name)
            return workflow.GeneratedViews(session_dir, "Показано 6 готових демо-ракурсів.")

    return generate_live_views(photo, seed)


def generate_live_views(photo: Path, seed: int) -> workflow.GeneratedViews:
    zero123_py = ROOT / "data" / "zero123_runtime" / "Scripts" / "python.exe"
    zero123_runner = ROOT / "data" / "zero123_task5_support" / "run_zero123.py"
    if sys.platform == "win32" and zero123_py.is_file() and zero123_runner.is_file():
        print(f"[*] Запуск локальної Zero123++ на NVIDIA GPU для нового фото: {photo.name} (seed: {seed})...", flush=True)
        backend = zero123_gen.Zero123SubprocessBackend(
            zero123_py,
            zero123_runner,
            vae_tiling=True,
            steps=int(os.environ.get("ZERO123_STEPS", "28")),
        )
        remover = rembg_mod.RembgRemover("u2netp")
        generator = zero123_gen.Zero123ViewGenerator(backend, remover, DEMO_STORAGE)
        directory = generator.generate_views(photo, seed)
        return workflow.GeneratedViews(directory, f"Zero123++ згенерував 6 ракурсів. Seed: {seed}.")

    raise workflow.GenerationUnavailable(
        "Жива генерація Zero123++ недоступна. Запустіть демо у Windows із налаштованим середовищем Zero123++."
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

    app = ui.build_app(generate_views_unified, reconstruct=demo_reconstruct_wrapper,
                       regenerate=generate_live_views)
    host = os.environ.get("GRADIO_SERVER_NAME", "127.0.0.1")
    app.launch(server_name=host, server_port=7860, share=False, inbrowser=host == "127.0.0.1",
               theme="soft", css=ui.CSS)


if __name__ == "__main__":
    main()
