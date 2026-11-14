"""Serve the direct photo-to-GLB workflow."""

import logging
import os
from pathlib import Path
import httpx
from tempfile import TemporaryDirectory

import gradio as gr
from gradio.processing_utils import save_file_to_cache
from gradio.utils import get_upload_folder

from gradio_compat import repair_model3d


FAST = "Fast (use Math)"
QUALITY = "Quality (use 3D model)"


def generate(photo, mirror_back=False, paint=False, mode=QUALITY):
    if not photo:
        raise gr.Error("Upload a photo first.")
    status = "Generating and texturing the model…" if paint else "Generating the model…"
    if mode == FAST:
        status = "Generating Wonder3D views and reconstructing the mesh…"
    yield None, None, status
    generator = os.environ.get("GENERATOR_URL", "http://127.0.0.1:8080").rstrip("/")
    if mode == FAST:
        generator = os.environ.get("FAST_GENERATOR_URL", "http://127.0.0.1:8081").rstrip("/")
    with TemporaryDirectory(prefix="photo-to-3d-") as directory:
        model = Path(directory) / "model.glb"
        try:
            with Path(photo).open("rb") as source, httpx.stream(
                "POST",
                f"{generator}/generate",
                params={"mirror_back": int(mirror_back and mode == QUALITY), "paint": int(paint and mode == QUALITY)},
                content=source,
                headers={"Content-Length": str(Path(photo).stat().st_size)},
                timeout=httpx.Timeout(5500, connect=5),
            ) as response:
                response.raise_for_status()
                with model.open("wb") as target:
                    for chunk in response.iter_bytes():
                        target.write(chunk)
            with model.open("rb") as result:
                if model.stat().st_size <= 12 or result.read(4) != b"glTF":
                    raise ValueError("Invalid GLB")
            cached_model = save_file_to_cache(model, get_upload_folder())
        except (OSError, ValueError, httpx.HTTPError):
            logging.exception("Photo-to-3D generation failed")
            yield None, None, "Generation failed. Check the server logs for details."
            return
        yield cached_model, cached_model, "Model ready."


EMPTY_STATUS = "Upload a photo of a single object."


def generate_preview(photo, mirror_back, paint, mode):
    for model, file, status in generate(photo, mirror_back, paint, mode):
        yield model, gr.update(value=file, interactive=file is not None), status


def clear_preview():
    return None, gr.update(value=None, interactive=False), EMPTY_STATUS


def build_app():
    repair_model3d()
    with gr.Blocks(title="3D Model Soft", analytics_enabled=False) as app:
        gr.Markdown("# 3D Model Soft", elem_id="workspace-title")
        with gr.Row(equal_height=True, elem_id="workspace"):
            with gr.Column(scale=1, min_width=270, elem_classes="workspace-panel"):
                gr.Markdown("## Input")
                photo = gr.Image(type="filepath", image_mode=None, sources=["upload"],
                                 label="Source photo", height=270)
                mode = gr.Radio([FAST, QUALITY], value=QUALITY, label="Generation mode")
                fast_help = gr.Markdown("Photo2Mesh uses Wonder3D views to reconstruct and texture the mesh.",
                            visible=False, elem_id="fast-help")
                mirror_back = gr.Checkbox(label="Mirror photo colors onto the back")
                paint = gr.Checkbox(label="Texture with Hunyuan Paint",
                                    value=os.environ.get("HUNYUAN_PAINT_DEFAULT") == "1")
                button = gr.Button("Generate model", variant="primary")
                status = gr.Markdown(EMPTY_STATUS, elem_id="generation-status")
            with gr.Column(scale=3, min_width=300, elem_classes="workspace-panel"):
                with gr.Row(equal_height=False, elem_id="preview-toolbar"):
                    gr.Markdown("## Preview")
                    download = gr.DownloadButton("Download GLB", size="sm", scale=0,
                                                 min_width=134, interactive=False)
                viewer = gr.Model3D(label="3D preview", show_label=False, height=490,
                                    interactive=False, clear_color=(0.92, 0.94, 0.96, 1))
        outputs = [viewer, download, status]
        generation = button.click(generate_preview, [photo, mirror_back, paint, mode], outputs,
                                  concurrency_limit=1, trigger_mode="once", api_visibility="private")
        photo.change(clear_preview, outputs=outputs, cancels=[generation],
                     queue=False, api_visibility="private")
        mode.change(
            lambda selected: (gr.update(visible=selected == QUALITY),
                              gr.update(visible=selected == FAST),
                              gr.update(visible=selected == QUALITY), *clear_preview()),
            inputs=mode, outputs=[paint, fast_help, mirror_back, *outputs], cancels=[generation],
            queue=False, api_visibility="private",
        )
    return app.queue(default_concurrency_limit=1)


if __name__ == "__main__":
    build_app().launch(
        server_name=os.environ.get("GRADIO_SERVER_NAME", "127.0.0.1"),
        server_port=int(os.environ.get("GRADIO_SERVER_PORT", "7860")),
        max_file_size="15mb",
        theme=gr.themes.Base(primary_hue=gr.themes.colors.blue,
                             neutral_hue=gr.themes.colors.slate,
                             font=["Inter", "system-ui", "sans-serif"]),
        css=Path(__file__).with_name("workspace.css").read_text(encoding="utf-8"),
    )
