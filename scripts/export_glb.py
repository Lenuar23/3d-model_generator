"""Export a saved NPZ mesh: python scripts/export_glb.py INPUT.npz OUTPUT.glb."""

import argparse
import importlib
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
Mesh = importlib.import_module("3d_model_soft.domain.models").Mesh
GlbExporter = importlib.import_module("3d_model_soft.infrastructure.export.glb_exporter").GlbExporter


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    try:
        with np.load(args.input, allow_pickle=False) as saved:
            mesh = Mesh(saved["vertices"], saved["faces"], saved["colors"] if "colors" in saved else None)
            output = GlbExporter().export(mesh, args.output)
    except (ValueError, KeyError, OSError) as error:
        parser.error(str(error))
    print(f"Exported GLB: {output.resolve()} ({output.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
