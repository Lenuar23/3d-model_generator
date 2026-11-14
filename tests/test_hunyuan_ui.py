"""Exercise the web service's HTTP boundary without loading GPU weights."""
from contextlib import contextmanager
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "services" / "web"))
import app as ui


class GenerationTests(unittest.TestCase):
    def test_missing_photo(self):
        with self.assertRaises(ui.gr.Error):
            list(ui.generate(None))

    def test_export_survives_temporary_cleanup_and_passes_options(self):
        payload = b"glTF" + bytes(20)

        @contextmanager
        def stream(method, url, **kwargs):
            self.assertEqual(method, "POST")
            self.assertTrue(url.endswith("/generate"))
            self.assertEqual(kwargs["params"], {"mirror_back": 1, "paint": 1})
            self.assertEqual(kwargs["content"].read(), b"photo")
            yield httpx.Response(200, content=payload, request=httpx.Request(method, url))

        with TemporaryDirectory() as directory:
            photo = Path(directory) / "photo.png"
            photo.write_bytes(b"photo")
            with patch.object(ui.httpx, "stream", side_effect=stream):
                results = list(ui.generate(photo, True, True))
        cached = Path(results[-1][0])
        self.assertEqual(cached.read_bytes(), payload)
        self.assertEqual(results[-1][0], results[-1][1])

    def test_fast_routes_to_photo2mesh_without_hunyuan_options(self):
        with TemporaryDirectory() as directory:
            photo = Path(directory) / "photo.png"
            photo.write_bytes(b"photo")
            with patch.dict(ui.os.environ, {"FAST_GENERATOR_URL": "http://fast:8080"}), \
                    patch.object(ui.httpx, "stream") as stream, self.assertLogs(level="ERROR"):
                stream.side_effect = httpx.ConnectError("test")
                list(ui.generate(photo, mirror_back=True, paint=True, mode=ui.FAST))
            self.assertEqual(stream.call_args.args, ("POST", "http://fast:8080/generate"))
            self.assertEqual(stream.call_args.kwargs["params"], {"mirror_back": 0, "paint": 0})

    def test_http_failure_and_invalid_output(self):
        for status, payload in [(500, b""), (200, b"not a GLB")]:
            with self.subTest(status=status), TemporaryDirectory() as directory:
                photo = Path(directory) / "photo.png"
                photo.write_bytes(b"photo")
                request = httpx.Request("POST", "http://generator/generate")
                response = httpx.Response(status, content=payload, request=request)

                @contextmanager
                def stream(*args, **kwargs):
                    yield response

                with patch.object(ui.httpx, "stream", side_effect=stream), self.assertLogs(level="ERROR"):
                    results = list(ui.generate(photo))
                self.assertEqual(results[-1][:2], (None, None))
