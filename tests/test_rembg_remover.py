"""Offline adapter tests: no rembg installation or model downloads required."""

import importlib
from pathlib import Path
import sys
import unittest
from unittest.mock import Mock, patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
adapter = importlib.import_module("3d_model_soft.infrastructure.ai.rembg_remover")
BackgroundRemover = importlib.import_module("3d_model_soft.domain.interfaces").BackgroundRemover


class RembgRemoverTests(unittest.TestCase):
    def setUp(self):
        self.backend = Mock()
        self.session = self.backend.new_session.return_value
        self.importer = patch.object(adapter.importlib, "import_module", return_value=self.backend)
        self.load = self.importer.start()
        self.addCleanup(self.importer.stop)

    def test_constructor_is_lazy_and_implements_protocol(self):
        remover = adapter.RembgRemover()
        self.assertIsInstance(remover, BackgroundRemover)
        self.load.assert_not_called()
        self.backend.new_session.assert_not_called()

    def test_valid_rgb_rgba_and_noncontiguous_inputs(self):
        for channels in (3, 4):
            image = np.arange(4 * 6 * channels, dtype=np.uint8).reshape(4, 6, channels)[:, ::2]
            image.setflags(write=False)
            output = np.full((4, 3, 4), 127, dtype=np.uint8)
            self.backend.remove.return_value = output
            result = adapter.RembgRemover("u2netp").remove(image)
            argument = self.backend.remove.call_args.args[0]
            np.testing.assert_array_equal(argument, image)
            self.assertFalse(np.shares_memory(argument, image))
            self.assertIs(self.backend.remove.call_args.kwargs["session"], self.session)
            np.testing.assert_array_equal(result, output)
            self.assertFalse(np.shares_memory(result, output))
            self.assertTrue(result.flags.writeable)

    def test_reuses_session_across_images(self):
        self.backend.remove.return_value = np.zeros((2, 3, 4), dtype=np.uint8)
        remover = adapter.RembgRemover("u2netp")
        for channels in (3, 4):
            remover.remove(np.zeros((2, 3, channels), dtype=np.uint8))
        self.load.assert_called_once_with("rembg")
        self.backend.new_session.assert_called_once_with(
            "u2netp", providers=["CPUExecutionProvider"],
        )

    def test_backend_mutation_does_not_change_input(self):
        image = np.full((2, 3, 4), 128, dtype=np.uint8)
        before = image.copy()

        def mutate(array, **kwargs):
            array[:] = 0
            return array

        self.backend.remove.side_effect = mutate
        result = adapter.RembgRemover().remove(image)
        np.testing.assert_array_equal(image, before)
        self.assertFalse(np.shares_memory(result, image))

    def test_invalid_images_fail_before_loading_backend(self):
        invalid = [None, "photo.png", [], np.zeros((2, 3, 3)),
                   np.zeros((2, 3), dtype=np.uint8), np.zeros((2, 3, 1), dtype=np.uint8),
                   np.zeros((2, 3, 5), dtype=np.uint8), np.zeros((0, 3, 3), dtype=np.uint8),
                   np.zeros((2, 0, 4), dtype=np.uint8)]
        for image in invalid:
            with self.subTest(image=type(image)):
                with self.assertRaisesRegex(ValueError, "uint8 RGB or RGBA"):
                    adapter.RembgRemover().remove(image)
        self.load.assert_not_called()

    def test_invalid_model_name(self):
        for name in (None, "", " ", 12):
            with self.subTest(name=name), self.assertRaises(ValueError):
                adapter.RembgRemover(name)
        self.load.assert_not_called()

    def test_missing_optional_dependency_has_install_guidance(self):
        for error in (ModuleNotFoundError("No rembg"), SystemExit(1)):
            self.load.side_effect = error
            with self.subTest(error=type(error)):
                with self.assertRaisesRegex(RuntimeError, "background-removal") as caught:
                    adapter.RembgRemover().remove(np.zeros((2, 3, 3), dtype=np.uint8))
                self.assertIs(caught.exception.__cause__, error)

    def test_session_failure_has_context_and_can_retry(self):
        error = OSError("Download failed")
        self.backend.new_session.side_effect = [error, self.session]
        self.backend.remove.return_value = np.zeros((2, 3, 4), dtype=np.uint8)
        remover = adapter.RembgRemover("u2netp")
        image = np.zeros((2, 3, 3), dtype=np.uint8)
        with self.assertRaisesRegex(RuntimeError, "u2netp") as caught:
            remover.remove(image)
        self.assertIs(caught.exception.__cause__, error)
        remover.remove(image)
        self.assertEqual(self.backend.new_session.call_count, 2)

    def test_inference_failure_keeps_input_and_original_cause(self):
        error = RuntimeError("Inference failed")
        self.backend.remove.side_effect = error
        image = np.full((2, 3, 3), 64, dtype=np.uint8)
        with self.assertRaisesRegex(RuntimeError, "rembg background removal failed") as caught:
            adapter.RembgRemover().remove(image)
        self.assertIs(caught.exception.__cause__, error)
        np.testing.assert_array_equal(image, np.full_like(image, 64))

    def test_invalid_backend_outputs_are_rejected(self):
        for output in (b"png", None, np.zeros((2, 3, 4)),
                       np.zeros((2, 3, 3), dtype=np.uint8),
                       np.zeros((3, 2, 4), dtype=np.uint8)):
            self.backend.remove.return_value = output
            with self.subTest(output=type(output)):
                with self.assertRaisesRegex(RuntimeError, "same-size uint8 RGBA"):
                    adapter.RembgRemover().remove(np.zeros((2, 3, 3), dtype=np.uint8))


if __name__ == "__main__":
    unittest.main()
