"""Offline tests for reproducible downloads and preservation of local files."""

import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("download_test_photos", ROOT / "scripts/download_test_photos.py")
downloader = importlib.util.module_from_spec(spec)
spec.loader.exec_module(downloader)


class DownloadTestPhotosTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.output = Path(self.folder.name)
        self.data = b"test photo bytes"
        self.photo = dict(filename="01_photo.jpg", download_url="https://example.invalid/photo.jpg",
                          bytes=len(self.data), sha256=hashlib.sha256(self.data).hexdigest())

    def test_real_manifest_has_ten_unique_licensed_photos_and_categories(self):
        photos = downloader.load_manifest()
        self.assertEqual(len(photos), 10)
        self.assertEqual({category: sum(p["category"] == category for p in photos)
                          for category in ("simple", "complex", "shiny", "transparent")},
                         dict(simple=3, complex=3, shiny=2, transparent=2))
        for photo in photos:
            self.assertTrue(photo["source_url"].startswith("https://commons.wikimedia.org/wiki/File:"))
            for field in ("author", "license", "license_url", "expected_challenge"):
                self.assertTrue(photo[field])

    def test_valid_download_and_idempotent_second_run(self):
        with patch.object(downloader.urllib.request, "urlopen", return_value=io.BytesIO(self.data)) as request:
            self.assertEqual(downloader.obtain_photo(self.photo, self.output), "downloaded")
            self.assertEqual(downloader.obtain_photo(self.photo, self.output), "verified")
        request.assert_called_once()
        self.assertEqual((self.output / self.photo["filename"]).read_bytes(), self.data)

    def test_modified_file_is_preserved_without_network(self):
        target = self.output / self.photo["filename"]
        target.write_bytes(b"user photo")
        with patch.object(downloader.urllib.request, "urlopen") as request:
            with self.assertRaises(FileExistsError):
                downloader.obtain_photo(self.photo, self.output)
        request.assert_not_called()
        self.assertEqual(target.read_bytes(), b"user photo")

    def test_corrupt_download_is_not_saved(self):
        with patch.object(downloader.urllib.request, "urlopen", return_value=io.BytesIO(b"changed")):
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                downloader.obtain_photo(self.photo, self.output)
        self.assertFalse((self.output / self.photo["filename"]).exists())

    def test_network_failure_does_not_create_file(self):
        with patch.object(downloader.urllib.request, "urlopen", side_effect=OSError("offline")):
            with self.assertRaises(OSError):
                downloader.obtain_photo(self.photo, self.output)
        self.assertFalse((self.output / self.photo["filename"]).exists())

    def test_check_only_does_not_download_missing_file(self):
        with patch.object(downloader.urllib.request, "urlopen") as request:
            with self.assertRaises(FileNotFoundError):
                downloader.obtain_photo(self.photo, self.output, check_only=True)
        request.assert_not_called()

    def test_manifest_rejects_path_traversal_duplicates_bad_hash_and_http(self):
        manifest = self.output / "manifest.json"
        invalid = [dict(self.photo, filename="../outside.jpg"), dict(self.photo, filename="C:\\outside.jpg"),
                   dict(self.photo, sha256="bad"), dict(self.photo, download_url="http://example.invalid/a"),
                   dict(self.photo, bytes=downloader.MAX_BYTES + 1)]
        for photo in invalid:
            manifest.write_text(json.dumps(dict(photos=[photo])), encoding="utf-8")
            with self.subTest(photo=photo), self.assertRaises(ValueError):
                downloader.load_manifest(manifest)
        manifest.write_text(json.dumps(dict(photos=[self.photo, self.photo])), encoding="utf-8")
        with self.assertRaises(ValueError):
            downloader.load_manifest(manifest)


if __name__ == "__main__":
    unittest.main()
