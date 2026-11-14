"""Internal HTTP service for GPU generation."""

import logging
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory
from urllib.parse import parse_qs, urlsplit

ROOT = Path(__file__).resolve().parent
MAX_PHOTO_BYTES = 15 * 1024 * 1024


class Handler(BaseHTTPRequestHandler):
    def generate(self, photo, model, options):
        command = [sys.executable, str(ROOT / "generate_hunyuan.py"), photo, model]
        if options.get("mirror_back") == ["1"]:
            command.append("--mirror-back")
        subprocess.run(command, check=True, timeout=1800)
        if options.get("paint") == ["1"]:
            output = model.parent / "paint"
            subprocess.run(
                [sys.executable, str(ROOT / "paint_multiview.py"), photo, model, output],
                check=True, timeout=3600,
            )
            model = output / "model.glb"
        return model

    def do_GET(self):
        if self.path == "/health":
            self.send_response(204)
            self.end_headers()
        else:
            self.send_error(404)

    def do_POST(self):
        url = urlsplit(self.path)
        if url.path != "/generate":
            self.send_error(404)
            return
        try:
            size = int(self.headers.get("Content-Length", ""))
        except ValueError:
            self.send_error(411)
            return
        if not 0 < size <= MAX_PHOTO_BYTES:
            self.send_error(413)
            return
        options = parse_qs(url.query)
        with TemporaryDirectory(prefix="photo-to-3d-") as directory:
            work = Path(directory)
            photo = work / "photo"
            model = work / "model.glb"
            try:
                self.connection.settimeout(30)
                payload = self.rfile.read(size)
                if len(payload) != size:
                    self.send_error(400)
                    return
                photo.write_bytes(payload)
                model = self.generate(photo, model, options)
                with model.open("rb") as result:
                    length = model.stat().st_size
                    if length <= 12 or result.read(4) != b"glTF":
                        raise ValueError("Invalid GLB")
                    result.seek(0)
                    self.send_response(200)
                    self.send_header("Content-Type", "model/gltf-binary")
                    self.send_header("Content-Length", str(length))
                    self.end_headers()
                    shutil.copyfileobj(result, self.wfile)
            except (OSError, subprocess.SubprocessError, ValueError):
                logging.exception("Generation failed")
                try:
                    self.send_error(500, "Generation failed")
                except OSError:
                    pass


if __name__ == "__main__":
    HTTPServer(("0.0.0.0", 8080), Handler).serve_forever()
