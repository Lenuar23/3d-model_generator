"""Generate a GLB directly from one photo."""

import argparse
from pathlib import Path

from photo_vertex_colors import apply_photo_colors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("photo", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--mirror-back", action="store_true")
    args = parser.parse_args()

    import torch
    from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline
    from PIL import Image, ImageOps
    from rembg import new_session, remove

    if not torch.cuda.is_available():
        raise RuntimeError("Hunyuan3D requires an available NVIDIA GPU with CUDA.")
    with Image.open(args.photo) as source:
        image = ImageOps.exif_transpose(source).convert("RGBA")
    image.thumbnail((1600, 1600))
    if image.getchannel("A").getextrema() == (255, 255):
        image = remove(image.convert("RGB"), session=new_session("u2netp"))
    if image.getchannel("A").getbbox() is None:
        raise ValueError("No foreground was found in the photo.")
    pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
        "tencent/Hunyuan3D-2mini", subfolder="hunyuan3d-dit-v2-mini",
        variant="fp16", device="cpu", dtype=torch.float16,
    )
    pipeline.to("cuda")
    mesh = pipeline(
        image=image, num_inference_steps=50, octree_resolution=256,
        num_chunks=4000, generator=torch.Generator(device="cuda").manual_seed(42),
    )[0]
    if mesh is None or len(mesh.faces) == 0:
        raise RuntimeError("Hunyuan3D returned no surface.")
    apply_photo_colors(mesh, image, mirror_back=args.mirror_back)
    mesh.export(args.output)


if __name__ == "__main__":
    main()
