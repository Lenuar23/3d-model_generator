"""Run pinned upstream Wonder3D image and normal inference in its own interpreter."""
import argparse
import json
import sys
from pathlib import Path
from time import perf_counter

SOURCE_REV = "d894f827aa8c2917761a0dad3ab40df74c7a5b24"
NAMES = ("front", "front_right", "right", "back", "left", "front_left")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--source", type=Path, default=Path("/opt/wonder-source"))
    parser.add_argument("--weights", type=Path, default=Path("models/wonder3d"))
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--guidance", type=float, default=3)
    args = parser.parse_args()
    import numpy as np
    import torch
    import diffusers
    from PIL import Image
    from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection

    args.output.mkdir(parents=True, exist_ok=True)
    if not torch.cuda.is_available():
        parser.error("CUDA is unavailable in the Wonder3D environment.")
    import subprocess
    revision = subprocess.check_output(["git", "-C", str(args.source), "rev-parse", "HEAD"], text=True).strip()
    if revision != SOURCE_REV:
        parser.error("Wonder3D source revision does not match the checkpoint-compatible upstream commit.")
    sys.path.insert(0, str(args.source.resolve()))
    from mvdiffusion.models.unet_mv2d_condition import UNetMV2DConditionModel
    from mvdiffusion.pipelines.pipeline_mvdiffusion_image import MVDiffusionImagePipeline

    start = perf_counter()
    dtype = torch.float16
    torch.cuda.reset_peak_memory_stats()
    weights = str(args.weights)
    clip = CLIPVisionModelWithProjection.from_pretrained(weights, subfolder="image_encoder",
                                                         torch_dtype=dtype, local_files_only=True)
    vae = diffusers.AutoencoderKL.from_pretrained(weights, subfolder="vae", torch_dtype=dtype,
                                                   local_files_only=True)
    unet = UNetMV2DConditionModel.from_pretrained(weights, subfolder="unet", torch_dtype=dtype,
                                                  local_files_only=True)
    pipe = MVDiffusionImagePipeline(
        image_encoder=clip, vae=vae, unet=unet,
        scheduler=diffusers.DDIMScheduler.from_pretrained(weights, subfolder="scheduler", local_files_only=True),
        feature_extractor=CLIPImageProcessor.from_pretrained(weights, subfolder="feature_extractor",
                                                              local_files_only=True),
        safety_checker=None, requires_safety_checker=False,
    ).to("cuda")
    vae.enable_slicing()
    # Adapts MIT-licensed Wonder3D attention (Copyright 2025 xxlong0).
    # Preserve upstream Q/K/V grouping while replacing explicit attention matrices with SDPA.
    for cls_name in ("MVAttnProcessor", "JointAttnProcessor"):
        import inspect
        from mvdiffusion.models import transformer_mv2d as module
        cls = getattr(module, cls_name)
        import textwrap
        code = textwrap.dedent(inspect.getsource(cls.__call__))
        if cls_name == "MVAttnProcessor":
            code = code.replace("multiview_attention=True",
                                "multiview_attention=True, sparse_mv_attention=False, mvcd_attention=False")
        before = "attention_probs = attn.get_attention_scores(query, key, attention_mask)\n    hidden_states = torch.bmm(attention_probs, value)"
        after = """hidden_states = torch.cat([
        F.scaled_dot_product_attention(q[:, None], key[:, None], value[:, None],
                                       attn_mask=attention_mask, scale=attn.scale)[:, 0]
        for q in query.split(256, dim=1)
    ], dim=1) if attention_mask is None else torch.bmm(
        attn.get_attention_scores(query, key, attention_mask), value)"""
        if before not in code:
            raise RuntimeError(f"Unexpected upstream attention implementation: {cls_name}")
        namespace = {}
        exec(compile(code.replace(before, after), "<Wonder3D SDPA adaptation>", "exec"), vars(module), namespace)
        cls.__call__ = namespace["__call__"]
    matrices = [np.loadtxt(args.source / f"mvdiffusion/data/fixed_poses/nine_views/000_{name}_RT.txt")
                for name in NAMES]
    centers = np.array([-m[:,:3].T @ m[:,3] for m in matrices])
    polar = np.arctan2(np.linalg.norm(centers[:,:2], axis=1), centers[:,2])
    heading = np.arctan2(centers[:,1], centers[:,0])
    embed = np.column_stack((np.zeros(6), polar-polar[0], (heading-heading[0])%(2*np.pi)))
    labels = np.concatenate([np.column_stack((embed,np.tile(task,(6,1))))
                             for task in ([1,0],[0,1])])
    with Image.open(args.image) as source:
        image = source.convert("RGB").resize((256,256))
    generator = torch.Generator(device="cuda").manual_seed(42)
    inference = perf_counter()
    with torch.inference_mode():
        result = pipe([image]*12, torch.tensor(labels, dtype=dtype), generator=generator,
                      guidance_scale=args.guidance, num_inference_steps=args.steps, eta=1).images
    torch.cuda.synchronize()
    if len(result) != 12:
        raise RuntimeError(f"Expected six normal/RGB pairs, received {len(result)} images.")
    for index in range(6):
        result[index].convert("RGB").save(args.output / f"normal-{index}.png")
        result[6+index].convert("RGB").save(args.output / f"rgb-{index}.png")
    report = {"source":revision,"names":NAMES,"matrices":[m.tolist() for m in matrices],
              "embedding":embed.tolist(),"seconds":perf_counter()-start,
              "inference_seconds":perf_counter()-inference,"steps":args.steps,"guidance":args.guidance,
              "torch":str(torch.__version__),"cuda":torch.version.cuda,"diffusers":diffusers.__version__,
              "allocated_mib":torch.cuda.max_memory_allocated()/2**20,
              "reserved_mib":torch.cuda.max_memory_reserved()/2**20}
    (args.output/"result.json").write_text(json.dumps(report,indent=2),encoding="utf-8")


if __name__ == "__main__":
    main()

