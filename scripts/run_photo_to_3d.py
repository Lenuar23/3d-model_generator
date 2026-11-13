"""Serve the direct photo-to-GLB workflow."""

import logging
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory

import gradio as gr
from gradio.processing_utils import save_file_to_cache
from gradio.utils import get_upload_folder

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def generate(photo, mirror_back=False, paint=False):
    if not photo:
        raise gr.Error("Спочатку завантажте фото.")
    yield None, None, "Будуємо 3D-модель… Перший запуск завантажує ваги."
    python = os.environ.get("HUNYUAN_PYTHON", sys.executable)
    with TemporaryDirectory(prefix="photo-to-3d-") as directory:
        model = Path(directory) / "model.glb"
        try:
            command = [python, str(ROOT / "scripts/generate_hunyuan.py"), photo, str(model)]
            if mirror_back:
                command.append("--mirror-back")
            subprocess.run(command, check=True, timeout=1800)
            if paint:
                yield None, None, "Текстуруємо модель за шістьма ракурсами…"
                output = Path(directory) / "paint"
                subprocess.run(
                    [python, str(ROOT / "scripts/paint_multiview.py"), photo, str(model), str(output)],
                    check=True, timeout=3600,
                )
                model = output / "model.glb"
            if not model.is_file() or model.stat().st_size <= 12:
                raise ValueError("Missing or incomplete GLB")
            # Gradio can serve a path already under GRADIO_TEMP_DIR as-is.
            # Publish a separate cached copy before deleting the working directory.
            cached_model = save_file_to_cache(model, get_upload_folder())
        except (OSError, ValueError, subprocess.SubprocessError):
            logging.exception("Photo-to-3D generation failed")
            yield None, None, "Генерація не вдалася. Перевірте фото та журнал запуску."
            return
        yield cached_model, cached_model, "Модель готова. Обертайте її або завантажте GLB."


def build_app():
    import importlib

    importlib.import_module("3d_model_soft.ui.gradio_compat").repair_model3d()
    with gr.Blocks(title="3D Model Soft", analytics_enabled=False) as app:
        gr.Markdown("# 3D Model Soft\nПеретворіть власне фото на 3D-модель.")
        with gr.Row():
            with gr.Column():
                photo = gr.Image(type="filepath", image_mode=None, sources=["upload"],
                                 label="Фото предмета", height=320)
                mirror_back = gr.Checkbox(label="Колір з обох боків (для округлих предметів)")
                paint = gr.Checkbox(label="Текстурувати з Hunyuan Paint",
                                    value=os.environ.get("HUNYUAN_PAINT_DEFAULT") == "1")
                button = gr.Button("Побудувати 3D-модель", variant="primary")
                status = gr.Markdown("Завантажте фото одного предмета.")
                download = gr.File(label="Завантажити GLB", interactive=False)
            viewer = gr.Model3D(label="3D-модель", height=480, interactive=False,
                                clear_color=(0.92, 0.94, 0.97, 1))
        outputs = [viewer, download, status]
        button.click(generate, [photo, mirror_back, paint], outputs, concurrency_limit=1,
                     trigger_mode="once", api_visibility="private")
        photo.clear(lambda: (None, None, "Завантажте фото одного предмета."), outputs=outputs)
    return app.queue(default_concurrency_limit=1)


if __name__ == "__main__":
    build_app().launch(
        server_name=os.environ.get("GRADIO_SERVER_NAME", "127.0.0.1"),
        server_port=int(os.environ.get("GRADIO_SERVER_PORT", "7860")),
        max_file_size="15mb", theme="soft",
    )
