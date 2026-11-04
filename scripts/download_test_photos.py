"""Fetch the licensed test photographs listed in docs/test_photos_manifest.json."""

import argparse
import hashlib
import json
from pathlib import Path
import re
import urllib.request


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "docs" / "test_photos_manifest.json"
MAX_BYTES = 20 * 1024 * 1024


def load_manifest(path=MANIFEST):
    photos = json.loads(Path(path).read_text(encoding="utf-8"))["photos"]
    names = set()
    for photo in photos:
        name = photo["filename"]
        if not re.fullmatch(r"[a-zA-Z0-9_-]+\.jpg", name) or name in names:
            raise ValueError(f"Unsafe or duplicate filename: {name}")
        names.add(name)
        if not re.fullmatch(r"[0-9a-f]{64}", photo["sha256"]):
            raise ValueError(f"Invalid SHA-256: {name}")
        if not isinstance(photo["bytes"], int) or not 0 < photo["bytes"] <= MAX_BYTES:
            raise ValueError(f"Invalid file size: {name}")
        if not photo["download_url"].startswith("https://"):
            raise ValueError(f"Expected an HTTPS download URL: {name}")
    return photos


def matches(data, photo):
    return len(data) == photo["bytes"] and hashlib.sha256(data).hexdigest() == photo["sha256"]


def obtain_photo(photo, output_dir, *, check_only=False):
    target = Path(output_dir) / photo["filename"]
    if target.is_symlink():
        raise FileExistsError(f"Refusing to follow an existing symlink: {target}")
    if target.exists():
        if not matches(target.read_bytes(), photo):
            raise FileExistsError(f"Existing file differs from the manifest; kept unchanged: {target}")
        return "verified"
    if check_only:
        raise FileNotFoundError(f"Missing test photograph: {target}")
    request = urllib.request.Request(
        photo["download_url"], headers={"User-Agent": "Photo3D-test-photo-downloader/0.1"},
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        data = response.read(MAX_BYTES + 1)
    if not matches(data, photo):
        raise ValueError(f"Download size or SHA-256 differs from the manifest: {photo['filename']}")
    target.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation prevents overwriting files created during the request.
    with target.open("xb") as stream:
        stream.write(data)
    return "downloaded"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data" / "test_photos")
    parser.add_argument("--check", action="store_true", help="Verify existing files without network access")
    args = parser.parse_args()
    try:
        for photo in load_manifest():
            result = obtain_photo(photo, args.output_dir, check_only=args.check)
            print(f"{result}: {photo['filename']}")
    except (OSError, ValueError, KeyError) as exc:
        parser.exit(1, f"{exc}\n")


if __name__ == "__main__":
    main()
