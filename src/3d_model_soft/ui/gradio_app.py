"""Local Gradio presentation for generating and confirming object views."""

from contextlib import ExitStack
from dataclasses import dataclass

from .presentation import CSS, translate
from functools import partial
import logging
from pathlib import Path
from shutil import copyfile
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING

from ..app.view_workflow import GenerationUnavailable, GenerateViews, Reconstruct, ViewSession, ViewWorkflow

if TYPE_CHECKING:
    from gradio import Blocks


_LOG = logging.getLogger(__name__)
_NO_RECONSTRUCTION = "Реконструкція поки недоступна."


@dataclass
class PresentationSession(ViewSession):
    language: str = "uk"
    status_text: str = "Завантажте фото предмета."
    previews: list | None = None


def build_app(generate: GenerateViews, reconstruct: Reconstruct | None = None, *,
              regenerate: GenerateViews | None = None) -> "Blocks":
    """Build an app with supplied application callbacks; the caller launches it."""
    try:
        import gradio as gr
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("Install the UI dependencies with uv sync --locked --extra ui") from exc

    workflow = ViewWorkflow(generate, reconstruct, regenerate=regenerate)
    from .gradio_compat import repair_model3d
    repair_model3d()

    def message(text):
        return text if workflow.can_reconstruct else f"{text} {_NO_RECONSTRUCTION}"

    def screen(session, status, *, previews=None, model=None):
        session.status_text = status
        if isinstance(previews, list) or previews is None:
            session.previews = previews
        if isinstance(previews, list):
            previews = [(image, translate(caption, session.language)) for image, caption in previews]
        status = translate(status, session.language)
        ready = session.photo is not None and not session.busy
        has_views = session.views_dir is not None
        return (
            previews, model, model, status, gr.update(interactive=not session.busy),
            gr.update(interactive=ready and not has_views),
            gr.update(interactive=ready and has_views),
            gr.update(interactive=ready and has_views and workflow.can_reconstruct),
        )

    def skip_screen():
        return tuple(gr.skip() for _ in outputs)

    def reset(upload, session):
        workflow.set_photo(session, upload)
        text = "Фото готове. Згенеруйте ракурси." if upload else "Завантажте фото предмета."
        return screen(session, message(text))

    def generate_action(upload, session, *, regenerate=False):
        request = workflow.begin_generation(session, upload, regenerate=regenerate)
        if request is None:
            yield skip_screen()
            return
        previous = gr.skip() if regenerate else None
        yield screen(session, "Готуємо ракурси…", previews=previous, model=previous)

        directory = None
        previews = []
        try:
            result, files = workflow.generate_with_description(request)
            for index, path in enumerate(files, start=1):
                with Image.open(path) as image:
                    if image.format != "PNG" or image.mode != "RGBA" or image.size != (320, 320):
                        raise ValueError("Expected a 320 × 320 RGBA PNG preview")
                    previews.append((image.copy(), f"Ракурс {index}"))
            directory = result.directory
            text = message(result.description)
        except GenerationUnavailable as exc:
            text = message(str(exc))
        except Exception:
            _LOG.exception("View generation failed")
            text = message("Не вдалося згенерувати ракурси. Перевірте фото або журнал запуску.")

        if workflow.finish_generation(session, request, directory):
            if directory is None:
                yield screen(session, text, previews=previous, model=previous)
            else:
                yield screen(session, text, previews=previews)
        else:
            yield skip_screen()

    def confirm_action(upload, session):
        request = workflow.begin_reconstruction(session, upload)
        if request is None:
            yield skip_screen()
            return
        yield screen(session, "Реконструюємо модель…", previews=gr.skip())

        with ExitStack() as stack:
            model = None
            try:
                result = workflow.reconstruct(request)
                # Gradio caches this individual verified output, not the source folder.
                temporary = Path(stack.enter_context(TemporaryDirectory(prefix="photo3d-glb-")))
                target = temporary / result.name
                copyfile(result, target)
                model = str(target)
                text = "Модель готова. Перегляньте її у 3D або завантажте GLB."
            except Exception:
                _LOG.exception("Reconstruction failed")
                text = "Не вдалося реконструювати модель. Перевірте журнал запуску або спробуйте інші ракурси."
            if workflow.finish_reconstruction(session, request):
                yield screen(session, text, previews=gr.skip(), model=model)
            else:
                yield skip_screen()

    with gr.Blocks(title="3D Model Soft", analytics_enabled=False) as demo:
        session = gr.State(PresentationSession())
        with gr.Row(elem_id="studio-header"):
            heading = gr.Markdown("# 3D Model Soft\nЗгенеруйте шість ракурсів предмета з одного фото.")
            language = gr.Dropdown(choices=[("Українська", "uk"), ("English", "en")],
                                   value="uk", label="Мова / Language", scale=0, min_width=170)

        with gr.Row():
            with gr.Column(scale=1, min_width=280, elem_classes="studio-card"):
                photo = gr.Image(label="Фото предмета", sources=["upload"], type="filepath",
                                 image_mode=None, height=300)
                generate_button = gr.Button("Згенерувати ракурси", variant="primary", interactive=False)
                with gr.Row():
                    regenerate_button = gr.Button("Перегенерувати", interactive=False)
                    confirm_button = gr.Button("Підтвердити", interactive=False)
                status = gr.Textbox(label="Стан", value=message("Завантажте фото предмета."),
                                    interactive=False, lines=2)
                download = gr.File(label="Завантажити GLB", interactive=False)
            with gr.Column(scale=2, min_width=320, elem_classes="studio-card"):
                viewer = gr.Model3D(label="Перегляд 3D-моделі", height=440, interactive=False, elem_id="model-preview",
                                    clear_color=(0.92, 0.94, 0.97, 1.0))
                hint = gr.Markdown("Обертайте модель мишею, наближайте колесом.")
                gallery = gr.Gallery(label="Ракурси", columns=3, rows=2, height=360,
                                     object_fit="contain", format="png", interactive=False)

        def change_language(code, state):
            state.language = code
            labels = ["Фото предмета", "Перегляд 3D-моделі", "Завантажити GLB"]
            updates = [gr.update(label=translate(label, code)) for label in labels]
            previews = state.previews
            if previews is not None:
                previews = [(image, translate(caption, code)) for image, caption in previews]
            return (
                "# 3D Model Soft\n" + translate("Згенеруйте шість ракурсів предмета з одного фото.", code),
                *updates,
                gr.update(value=translate("Згенерувати ракурси", code)),
                gr.update(value=translate("Перегенерувати", code)),
                gr.update(value=translate("Підтвердити", code)),
                gr.update(label=translate("Стан", code), value=translate(state.status_text, code)),
                gr.update(label=translate("Ракурси", code), value=previews),
                translate("Обертайте модель мишею, наближайте колесом.", code),
            )

        language.change(change_language, inputs=[language, session],
                        outputs=[heading, photo, viewer, download, generate_button,
                                 regenerate_button, confirm_button, status, gallery, hint],
                        queue=False, api_visibility="private")
        inputs = [photo, session]
        outputs = [gallery, viewer, download, status, photo,
                   generate_button, regenerate_button, confirm_button]
        gr.on([photo.upload, photo.clear], fn=reset, inputs=inputs, outputs=outputs,
              queue=False, api_name="reset", api_visibility="private")
        work = dict(inputs=inputs, outputs=outputs, concurrency_id="photo3d-work",
                    concurrency_limit=1, api_visibility="private")
        generate_button.click(generate_action, api_name="generate", **work)
        regenerate_button.click(partial(generate_action, regenerate=True), api_name="regenerate", **work)
        if workflow.can_reconstruct:
            confirm_button.click(confirm_action, api_name="confirm", **work)
    return demo.queue(default_concurrency_limit=1)
