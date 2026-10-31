"""Generate an analytic fixture: python scripts/generate_synthetic.py OUTPUT."""

import argparse
import importlib
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
generate_dataset = importlib.import_module(
    "3d_model_soft.infrastructure.synthetic_dataset"
).generate_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--radii", type=float, nargs=3, default=(0.6, 0.8, 0.5))
    parser.add_argument("--version", choices=("v1.1", "v1.2"), default="v1.2")
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--resolution", type=int, default=64)
    args = parser.parse_args()
    try:
        output = generate_dataset(args.output, radii=tuple(args.radii), version=args.version,
                                  size=args.size, resolution=args.resolution)
    except (ValueError, FileExistsError) as error:
        parser.error(str(error))
    print(f"Generated synthetic dataset: {output.resolve()}")


if __name__ == "__main__":
    main()
