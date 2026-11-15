"""Regression tests for the generator API and both generation paths."""
from io import BytesIO
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

SERVICES = Path(__file__).resolve().parents[1] / "services"


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


server = load_module("generation_server", SERVICES / "generator" / "server.py")


class GeneratorHTTPTests(unittest.TestCase):
    def request(self, method, path, body=b"", headers=None):
        headers = {"Content-Length": str(len(body)), **(headers or {})}
        head = [f"{method} {path} HTTP/1.1", *(f"{k}: {v}" for k, v in headers.items())]
        wire = ("\r\n".join(head) + "\r\n\r\n").encode("ascii") + body
        connection = Mock()
        connection.makefile.return_value = BytesIO(wire)
        server.Handler(connection, ("127.0.0.1", 12345), None)
        response = b"".join(call.args[0] for call in connection.sendall.call_args_list)
        head, payload = response.split(b"\r\n\r\n", 1)
        return int(head.split()[1]), payload

    def test_health_unknown_route_and_upload_limits(self):
        self.assertEqual(self.request("GET", "/health"), (204, b""))
        self.assertEqual(self.request("GET", "/unknown")[0], 404)
        self.assertEqual(self.request("POST", "/generate", b"")[0], 413)
        headers = {"Content-Length": str(server.MAX_PHOTO_BYTES + 1)}
        self.assertEqual(self.request("POST", "/generate", headers=headers)[0], 413)

    def test_upload_options_and_download(self):
        payload = b"glTF" + bytes(20)
        workspaces = []

        def generate(photo, model, options):
            self.assertEqual(photo.read_bytes(), b"source photo")
            self.assertEqual(options, {"mirror_back": ["1"], "paint": ["0"]})
            workspaces.append(model.parent)
            model.write_bytes(payload)
            return model

        with patch.object(server.Handler, "generate", side_effect=generate):
            self.assertEqual(self.request("POST", "/generate?mirror_back=1&paint=0", b"source photo"),
                             (200, payload))
        self.assertFalse(workspaces[0].exists())

    def test_invalid_glb_is_rejected(self):
        def generate(photo, model, options):
            model.write_bytes(bytes(32))
            return model

        with patch.object(server.Handler, "generate", side_effect=generate), self.assertLogs(level="ERROR"):
            self.assertEqual(self.request("POST", "/generate", b"photo")[0], 500)


class QualityTests(unittest.TestCase):
    def test_shape_then_optional_paint(self):
        photo = Path("request/photo")
        model = photo.with_name("model.glb")
        for paint in (False, True):
            with self.subTest(paint=paint), patch.object(server.subprocess, "run") as run:
                result = server.Handler.generate(None, photo, model, {
                    "mirror_back": ["1"], "paint": [str(int(paint))],
                })
                shape = run.call_args_list[0]
                self.assertEqual(shape.args[0][2:], [photo, model, "--mirror-back"])
                self.assertEqual(shape.kwargs, {"check": True, "timeout": 1800})
                self.assertEqual(run.call_count, 2 if paint else 1)
                if paint:
                    texture = run.call_args_list[1]
                    self.assertEqual(texture.args[0][2:], [photo, model, model.parent / "paint"])
                    self.assertEqual(texture.kwargs, {"check": True, "timeout": 3600})
                self.assertEqual(result, model.parent / "paint/model.glb" if paint else model)


class HullValidationError(ValueError):
    pass


class FastTests(unittest.TestCase):
    def setUp(self):
        self.views = ("front", "right")
        self.ensure_models = Mock()
        self.generate_views = Mock(return_value=(Path("cached-views"), {}))
        self.reconstruct = Mock()
        modules = {
            "server": server,
            "models": SimpleNamespace(ensure_models=self.ensure_models),
            "photo2mesh.multiview": SimpleNamespace(HullValidationError=HullValidationError),
            "photo2mesh.wonder3d_adapter": SimpleNamespace(
                generate_wonder=self.generate_views,
                reconstruct_wonder=self.reconstruct,
            ),
        }
        with patch.dict(sys.modules, modules):
            self.fast = load_module("fast_handler", SERVICES / "fast" / "fast_server.py")
        self.photo = Path("request/photo")
        self.output = self.photo.with_name("model.glb")
        self.result = (SimpleNamespace(exports=SimpleNamespace(glb=self.output)), [], [], [])

    def generate(self):
        return self.fast.FastHandler.generate(None, self.photo, self.output, {})

    def test_normal_path(self):
        self.reconstruct.return_value = self.result
        self.assertEqual(self.generate(), self.output)
        self.ensure_models.assert_called_once_with()
        self.generate_views.assert_called_once_with(self.photo, folder=self.output.parent)
        self.reconstruct.assert_called_once_with(
            self.photo, resolution=64, reuse=Path("cached-views"), excluded=(),
        )

    def test_retries_resolutions_then_omits_a_view_without_regenerating(self):
        self.reconstruct.side_effect = [HullValidationError()] * 5 + [["right", "front"], self.result]
        with self.assertLogs(level="WARNING"):
            self.assertEqual(self.generate(), self.output)
        self.generate_views.assert_called_once()
        self.assertEqual(
            [(c.kwargs["resolution"], c.kwargs["excluded"]) for c in self.reconstruct.call_args_list
             if not (c.kwargs.get("rank_only") or c.kwargs.get("weighted"))],
            [(64, ()), (80, ()), (96, ()), (48, ()), (64, ("right",))],
        )
        for attempt in self.reconstruct.call_args_list:
            self.assertEqual(attempt.kwargs["reuse"], Path("cached-views"))

    def test_weighted_reconstruction_can_keep_all_views(self):
        self.reconstruct.side_effect = [HullValidationError()] * 4 + [self.result]
        with self.assertLogs(level="WARNING"):
            self.assertEqual(self.generate(), self.output)
        self.reconstruct.assert_called_with(self.photo, reuse=Path("cached-views"), weighted=True)
        self.generate_views.assert_called_once()

    def test_exhausted_attempts_surface_the_failure(self):
        error = HullValidationError("invalid hull")
        self.reconstruct.side_effect = [error] * 5 + [list(self.views)] + [error] * len(self.views)
        with self.assertLogs(level="WARNING"), self.assertRaisesRegex(HullValidationError, "invalid hull"):
            self.generate()
        self.assertEqual(self.reconstruct.call_count, 6 + len(self.views))

    def test_unrelated_error_is_not_retried(self):
        self.reconstruct.side_effect = OSError("cannot write export")
        with self.assertRaisesRegex(OSError, "cannot write export"):
            self.generate()
        self.assertEqual(self.reconstruct.call_count, 1)
