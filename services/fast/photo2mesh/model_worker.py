"""Run local CLIP recognition in its isolated interpreter and write a JSON result."""
import argparse
import json
from pathlib import Path

from PIL import Image


def recognize(args):
    """Classify images with the same CPU adapter used by the application."""
    from ai_assist import LocalClip

    recognizer = LocalClip(args.model)
    results = []
    for path in args.images:
        with Image.open(path) as image:
            category, margin = recognizer.classify(image)
        results.append({"image": str(path), "category": category, "logit_margin": margin})
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=["recognize"])
    parser.add_argument("images", type=Path, nargs="+")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    result = recognize(args)
    (args.output / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
