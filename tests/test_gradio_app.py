"""Session workflow tests and focused Gradio event integration checks."""

import base64
import importlib
from pathlib import Path
import struct
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
workflow_module = importlib.import_module("3d_model_soft.app.view_workflow")
ui = importlib.import_module("3d_model_soft.ui.gradio_app")


class WorkflowTests(unittest.TestCase):
    def test_photo_change_invalidates_inflight_generation_and_keeps_sessions_independent(self):
        workflow = workflow_module.ViewWorkflow(Mock(), Mock())
        first = workflow_module.ViewSession()
        second = workflow_module.ViewSession()
        photo = Path("one.jpg")
        workflow.set_photo(first, photo)
        request = workflow.begin_generation(first, photo)
        self.assertIsNotNone(request)
        self.assertIsNone(workflow.begin_generation(first, photo))
        self.assertIsNone(workflow.begin_reconstruction(first, photo))
        self.assertIsNone(workflow.begin_generation(second, photo))
        workflow.set_photo(first, Path("two.jpg"))
        self.assertFalse(workflow.finish_generation(first, request, Path("old/views")))
        self.assertIsNone(first.views_dir)
        self.assertIsNone(workflow.begin_reconstruction(first, photo))
        self.assertIsNone(second.photo)

    def test_regeneration_changes_seed_and_confirmation_uses_latest_bundle(self):
        reconstruct = Mock()
        workflow = workflow_module.ViewWorkflow(Mock(), reconstruct)
        session = workflow_module.ViewSession()
        photo = Path("photo.jpg")
        workflow.set_photo(session, photo)
        with patch.object(workflow_module, "randbits", return_value=42):
            original = workflow.begin_generation(session, photo)
            workflow.finish_generation(session, original, Path("original/views"))
            regenerated = workflow.begin_generation(session, photo, regenerate=True)
        self.assertNotEqual(regenerated.seed, original.seed)
        self.assertIsNone(workflow.begin_reconstruction(session, photo))
        latest = Path("latest/views")
        workflow.finish_generation(session, regenerated, latest)
        request = workflow.begin_reconstruction(session, photo)
        self.assertEqual(request.views_dir, latest)
        with tempfile.TemporaryDirectory() as folder:
            model = Path(folder) / "model.glb"
            model.write_bytes(b"glTF" + struct.pack("<II", 2, 12))
            reconstruct.return_value = model
            self.assertEqual(workflow.reconstruct(request), model.resolve())
        reconstruct.assert_called_once_with(latest)

    def test_invalid_model_file_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            model = Path(folder) / "wrong.glb"
            workflow = workflow_module.ViewWorkflow(Mock(), lambda directory: model)
            request = workflow_module.ReconstructionRequest(Path("views"), 1)
            malformed = (
                b"glTFgarbage",
                b"wrong format",
                b"glTF" + struct.pack("<II", 1, 12),
                b"glTF" + struct.pack("<II", 2, 100),
                b"glTF" + struct.pack("<II", 2, 12) + b"extra bytes",
            )
            for contents in malformed:
                with self.subTest(contents=contents):
                    model.write_bytes(contents)
                    with self.assertRaisesRegex(ValueError, "GLB header"):
                        workflow.reconstruct(request)


@unittest.skipUnless(importlib.util.find_spec("gradio"), "Install the optional ui extra")
class GradioEventTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        from PIL import Image

        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.root = Path(folder.name)
        self.photo = self.root / "photo.png"
        Image.new("RGB", (40, 40), (160, 70, 20)).save(self.photo)
        self.views = self.root / "views"
        self.views.mkdir()
        (self.views / "cameras.json").write_text("{}", encoding="utf-8")
        for index in range(6):
            Image.new("RGBA", (320, 320), (20 * index, 100, 60, 128)).save(self.views / f"view_{index:02d}.png")

    def handlers(self, app):
        self.addCleanup(app.close, verbose=False)
        return {event.api_name: event for event in app.fns.values()}

    async def test_gallery_regenerate_and_unavailable_confirmation(self):
        generate = Mock(return_value=self.views)
        events = self.handlers(ui.build_app(generate))
        self.assertNotIn("confirm", events)
        self.assertFalse(events["reset"].queue)
        self.assertEqual(events["generate"].concurrency_id, events["regenerate"].concurrency_id)
        self.assertEqual(events["generate"].concurrency_limit, 1)

        session = workflow_module.ViewSession()
        start = events["reset"].fn(str(self.photo), session)
        self.assertTrue(start[4]["interactive"])
        self.assertFalse(start[6]["interactive"])
        busy, done = list(events["generate"].fn(str(self.photo), session))
        self.assertFalse(busy[3]["interactive"])
        self.assertEqual(len(done[0]), 6)
        self.assertEqual(done[0][0][0].mode, "RGBA")
        self.assertEqual(done[0][0][0].getpixel((0, 0))[3], 128)
        self.assertTrue(done[5]["interactive"])
        self.assertFalse(done[6]["interactive"])
        self.assertIn("Реконструкція поки недоступна", done[2])
        list(events["regenerate"].fn(str(self.photo), session))
        self.assertNotEqual(generate.call_args_list[0].args[1], generate.call_args_list[1].args[1])

    async def test_upload_reset_discards_old_results_and_failure_clears_preview(self):
        generate = Mock(return_value=self.views)
        events = self.handlers(ui.build_app(generate, Mock()))
        session = workflow_module.ViewSession()
        events["reset"].fn(str(self.photo), session)
        old = events["generate"].fn(str(self.photo), session)
        next(old)
        updated_photo = str(self.root / "new.png")
        cleared = events["reset"].fn(updated_photo, session)
        self.assertIsNone(cleared[0])
        self.assertIsNone(cleared[1])
        self.assertFalse(cleared[6]["interactive"])
        stale = next(old)
        self.assertTrue(all(update == {"__type__": "update"} for update in stale))
        old.close()
        self.assertIsNone(session.views_dir)

        (self.views / "view_05.png").write_bytes(b"bad image")
        with self.assertLogs(ui.__name__, level="ERROR"):
            _, failed = list(events["generate"].fn(updated_photo, session))
        self.assertIsNone(failed[0])
        self.assertFalse(failed[6]["interactive"])
        self.assertIn("Не вдалося згенерувати", failed[2])

    async def test_confirm_stages_only_the_returned_glb(self):
        model = self.root / "model.glb"
        model.write_bytes(b"glTF" + struct.pack("<II", 2, 12))
        reconstruct = Mock(return_value=model)
        events = self.handlers(ui.build_app(Mock(return_value=self.views), reconstruct))
        session = workflow_module.ViewSession()
        events["reset"].fn(str(self.photo), session)
        _, done = list(events["generate"].fn(str(self.photo), session))
        self.assertTrue(done[6]["interactive"])

        confirm = events["confirm"].fn(str(self.photo), session)
        try:
            busy = next(confirm)
            self.assertIsNone(busy[1])
            ready = next(confirm)
            staged = Path(ready[1])
            self.assertNotEqual(staged, model)
            self.assertEqual(staged.read_bytes(), model.read_bytes())
            self.assertEqual(list(staged.parent.iterdir()), [staged])
        finally:
            confirm.close()
        reconstruct.assert_called_once_with(self.views.resolve())
        self.assertFalse(staged.exists())

        reconstruct.side_effect = RuntimeError("test reconstruction error")
        with self.assertLogs(ui.__name__, level="ERROR"):
            busy, failed = list(events["confirm"].fn(str(self.photo), session))
        self.assertIsNone(busy[1])
        self.assertIsNone(failed[1])
        self.assertIn("Не вдалося реконструювати", failed[2])
        self.assertTrue(failed[6]["interactive"])

    async def test_process_api_caches_gallery_and_glb_before_temporary_cleanup(self):
        from gradio.state_holder import SessionState

        model = self.root / "source-model.glb"
        model.write_bytes(b"glTF" + struct.pack("<II", 2, 12))
        reconstruct = Mock(return_value=model)
        app = ui.build_app(Mock(return_value=self.views), reconstruct)
        events = self.handlers(app)
        state = SessionState(app)
        session_component = events["reset"].inputs[1]
        other_state = SessionState(app)
        self.assertIsNot(state[session_component._id], other_state[session_component._id])
        payload = {
            "url": "data:image/png;base64," + base64.b64encode(self.photo.read_bytes()).decode(),
            "meta": {"_type": "gradio.FileData"},
        }
        inputs = [payload, None]
        await app.process_api(events["reset"], inputs, state=state)
        busy = await app.process_api(events["generate"], inputs, state=state)
        generated = await app.process_api(events["generate"], [], state=state, iterator=busy["iterator"])
        previews = generated["data"][0]
        self.assertEqual(len(previews), 6)
        self.assertTrue(all(Path(item["image"]["path"]).is_file() for item in previews))
        await app.process_api(events["generate"], [], state=state, iterator=generated["iterator"])

        busy = await app.process_api(events["confirm"], inputs, state=state)
        reconstructed = await app.process_api(events["confirm"], [], state=state, iterator=busy["iterator"])
        cached_path = Path(reconstructed["data"][1]["path"])
        self.assertNotEqual(cached_path, model)
        self.assertEqual(cached_path.read_bytes(), model.read_bytes())
        await app.process_api(events["confirm"], [], state=state, iterator=reconstructed["iterator"])
        self.assertTrue(cached_path.is_file())
        self.assertEqual(cached_path.read_bytes(), model.read_bytes())
        reconstruct.assert_called_once_with(self.views.resolve())


if __name__ == "__main__":
    unittest.main()
