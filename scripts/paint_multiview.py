"""Texture an existing Hunyuan mesh using geometry-conditioned multi-view Paint."""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "data/hunyuan_source"))
os.environ.setdefault("HF_HOME", str(ROOT / "data/hunyuan_cache"))
os.environ.setdefault("U2NET_HOME", str(ROOT / "data/u2net"))
os.environ.setdefault("NUMBA_DISABLE_JIT", "1")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("photo", type=Path)
    parser.add_argument("mesh", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--weights", type=Path, default=ROOT / "data/hunyuan_paint_weights")
    parser.add_argument("--view-size", type=int, choices=(512,), default=512)
    parser.add_argument("--max-faces", type=int, default=60000)
    args = parser.parse_args()
    if not args.photo.is_file() or not args.mesh.is_file() or args.max_faces < 4:
        parser.error("Existing photo/mesh and max-faces >= 4 required")
    import torch
    import trimesh
    from PIL import Image, ImageOps
    from rembg import remove, new_session
    from hy3dgen.texgen.pipelines import Hunyuan3DPaintPipeline, Hunyuan3DTexGenConfig
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required")
    if not args.weights.exists() or os.environ.get("HUNYUAN_PAINT_DOWNLOAD") == "1":
        from huggingface_hub import snapshot_download
        snapshot_download(
            "tencent/Hunyuan3D-2",
            revision="9cd649ba6913f7a852e3286bad86bfa9a2d83dcf",
            allow_patterns=["hunyuan3d-delight-v2-0/*", "hunyuan3d-paint-v2-0-turbo/*"],
            local_dir=args.weights,
            max_workers=2,
        )
    args.output.mkdir(parents=True, exist_ok=False)
    with Image.open(args.photo) as source:
        image = ImageOps.exif_transpose(source).convert("RGBA")
    image.thumbnail((1024, 1024))
    if image.getchannel("A").getextrema() == (255, 255):
        image = remove(image.convert("RGB"), session=new_session("u2netp"))
    if image.getchannel("A").getbbox() is None:
        raise ValueError("No foreground found")
    mesh = trimesh.load(args.mesh, force="mesh", process=False)
    source_faces = len(mesh.faces)
    if source_faces > args.max_faces:
        mesh = mesh.simplify_quadric_decimation(face_count=args.max_faces)
    # Existing vertex colors must not compete with the new UV texture.
    mesh.visual = trimesh.visual.texture.TextureVisuals()
    config = Hunyuan3DTexGenConfig(
        str(args.weights / "hunyuan3d-delight-v2-0"),
        str(args.weights / "hunyuan3d-paint-v2-0-turbo"),
        "hunyuan3d-paint-v2-0-turbo",
    )
    config.device = "cpu"  # Do not load both diffusion networks onto the 6 GB GPU.
    config.render_size = 512
    config.texture_size = 1024
    print("Loading Paint networks on CPU...", flush=True)
    pipeline = Hunyuan3DPaintPipeline(config)
    for key in ("delight_model", "multiview_model"):
        network = pipeline.models[key].pipeline
        network.enable_model_cpu_offload()
        network.enable_vae_slicing()
        network.enable_vae_tiling()
    multiview = pipeline.models["multiview_model"]
    multiview.view_size = args.view_size

    class CaptureViews:
        def __call__(self, *inputs, **kwargs):
            print("Generating six geometry-conditioned views...", flush=True)
            views = multiview(*inputs, **kwargs)
            for index, view in enumerate(views):
                view.save(args.output / f"view_{index:02d}.png")
            return views

    pipeline.models["multiview_model"] = CaptureViews()
    torch.cuda.reset_peak_memory_stats()
    print("Removing lighting, rendering geometry and painting...", flush=True)
    result = pipeline(mesh, image)
    result.export(args.output / "model.glb")
    report = dict(source_faces=source_faces, output_faces=len(result.faces),
                  view_size=args.view_size, texture_size=config.texture_size,
                  peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
                  camera_azimuths=config.candidate_camera_azims,
                  camera_elevations=config.candidate_camera_elevs)
    (args.output / "report.json").write_text(json.dumps(report, indent=2))
    print(f"Saved {args.output / 'model.glb'}", flush=True)


if __name__ == "__main__":
    main()
