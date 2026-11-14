"""Experimental local photo caption -> material image -> UV-textured GLB.

Uses the existing mesh; hidden surfaces are synthesized, not recovered.
"""
import argparse
import gc
import json
from pathlib import Path


def describe(image):
    import torch
    from transformers import BlipProcessor, BlipForConditionalGeneration
    model_id = "Salesforce/blip-image-captioning-base"
    processor = BlipProcessor.from_pretrained(model_id)
    model = BlipForConditionalGeneration.from_pretrained(model_id).eval()
    with torch.inference_mode():
        tokens = model.generate(**processor(image.convert("RGB"), return_tensors="pt"),
                                max_new_tokens=60, num_beams=3)
    caption = processor.decode(tokens[0], skip_special_tokens=True)
    del model, processor
    gc.collect()
    return caption


def generate_material(prompt, seed, steps, reference):
    import torch
    from diffusers import StableDiffusionImg2ImgPipeline
    if not torch.cuda.is_available():
        raise RuntimeError("Texture generation requires CUDA.")
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        "stable-diffusion-v1-5/stable-diffusion-v1-5",
        torch_dtype=torch.float16, variant="fp16", use_safetensors=True,
    )
    pipe.enable_model_cpu_offload()
    pipe.enable_vae_slicing()
    result = pipe(
        prompt, image=reference.resize((512, 512)), strength=.45,
        negative_prompt="whole object, leaves, stem, silhouette, background, text, watermark, shadow, glare",
        num_inference_steps=steps,
        generator=torch.Generator(device="cpu").manual_seed(seed),
    )
    if result.nsfw_content_detected and any(result.nsfw_content_detected):
        raise RuntimeError("Material generation was rejected by the model safety checker.")
    image = result.images[0]
    del result, pipe
    gc.collect()
    torch.cuda.empty_cache()
    return image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("photo", type=Path)
    parser.add_argument("mesh", type=Path)
    parser.add_argument("output", type=Path, help="New output directory; never overwrites input")
    parser.add_argument("--material", help="Optional English material description, replaces inferred prompt")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=25)
    parser.add_argument("--size", type=int, choices=(512, 1024, 2048), default=1024)
    args = parser.parse_args()
    if not args.photo.is_file() or not args.mesh.is_file():
        parser.error("Photo and mesh must exist")
    if args.steps < 3:
        parser.error("steps must be at least 3 for strength=0.45")
    args.output.mkdir(parents=True, exist_ok=False)
    from PIL import Image, ImageOps
    from rembg import remove, new_session
    import trimesh
    from texture_bake import bake_texture
    with Image.open(args.photo) as source:
        image = ImageOps.exif_transpose(source).convert("RGBA")
    image.thumbnail((1024, 1024))
    if image.getchannel("A").getextrema() == (255, 255):
        image = remove(image.convert("RGB"), session=new_session("u2netp"))
    if image.getchannel("A").getbbox() is None:
        raise ValueError("No foreground")
    image.save(args.output / "foreground.png")
    print("Describing photo on CPU...", flush=True)
    # Neutral background avoids interpreting transparent pixels as a black object.
    reference = Image.new("RGB", image.size, "white")
    reference.paste(image, mask=image.getchannel("A"))
    caption = describe(reference)
    prompt = args.material or (
        f"Extreme macro photograph of the surface material of {caption}. "
        "Flat uniform surface texture filling the entire frame, fine natural details, "
        "even diffuse lighting, no entire object, no background."
    )
    report = dict(caption=caption, material_prompt=prompt, seed=args.seed, steps=args.steps,
                  limitation="Hidden material is an inference. Single-photo projection assumes front +Z.",
                  caption_model="Salesforce/blip-image-captioning-base",
                  material_model="stable-diffusion-v1-5/stable-diffusion-v1-5",
                  strength=.45, texture_size=args.size, projection="front +Z, triplanar hidden material")
    (args.output / "description.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Caption: {caption}\nGenerating material on CUDA...", flush=True)
    # Anchor generation to a real surface patch instead of asking for an entire object.
    import numpy as np
    from scipy.ndimage import distance_transform_edt
    rgba = np.asarray(image)
    foreground = rgba[..., 3] >= 128
    median = np.median(rgba[..., :3][foreground], axis=0)
    similarity = np.linalg.norm(rgba[..., :3].astype(float)-median, axis=2)
    interior = distance_transform_edt(foreground & (similarity < 80))
    cy, cx = np.unravel_index(interior.argmax(), interior.shape)
    radius = int(interior[cy, cx] / 1.5)
    if radius < 4:
        raise ValueError("No sufficiently large uniform material patch; use a closer photo")
    patch = image.crop((cx-radius, cy-radius, cx+radius, cy+radius)).convert("RGB")
    patch.save(args.output / "material_reference.png")
    material = generate_material(prompt, args.seed, args.steps, patch)
    material.save(args.output / "material.png")
    print("Unwrapping mesh and baking photo + inferred material...", flush=True)
    mesh = trimesh.load(args.mesh, force="mesh", process=False)
    textured, atlas = bake_texture(mesh, image, material, size=args.size)
    atlas.save(args.output / "texture.png")
    textured.export(args.output / "model.glb")
    print(f"Saved {args.output / 'model.glb'}", flush=True)


if __name__ == "__main__":
    main()
