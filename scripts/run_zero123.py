"""Generate the original six-view Zero123++ v1.2 grid on a CUDA GPU."""

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from urllib.request import urlopen


MODEL_ID = "sudo-ai/zero123plus-v1.2"
MODEL_REVISION = "2da07e89919e1a130c9b5add1584c70c7aa065fd"
PIPELINE_ID = "sudo-ai/zero123plus-pipeline"
PIPELINE_REVISION = "983e66d28a3637ddd8e3e2fd8165cdff32230872"
DATASET_REVISION = "e2f81456de75b15aec6d4839bb1ecb9c44c73177"
DATASET_ROOT = f"https://raw.githubusercontent.com/Lenuar23/3d-model_generator/{DATASET_REVISION}"
MASK_MODEL = "u2netp"
INPUT_SIZE = 320
GRID_SIZE = (640, 960)
GRAY = (127, 127, 127)


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load_batch(photos_dir):
    with urlopen(f"{DATASET_ROOT}/docs/test_photos_manifest.json", timeout=30) as response:
        photos = json.load(response)["photos"]
    filenames = [photo["filename"] for photo in photos]
    if len(filenames) != 10 or len(set(filenames)) != 10:
        raise ValueError("Expected the ten distinct photographs in the pinned manifest")
    inputs = []
    for photo in photos:
        name = photo["filename"]
        if not re.fullmatch(r"[a-zA-Z0-9_-]+\.jpg", name):
            raise ValueError(f"Unexpected manifest filename: {name}")
        path = photos_dir / name
        if not path.is_file() or sha256(path) != photo["sha256"]:
            raise ValueError(f"Missing or changed test photograph: {path}")
        inputs.append((path, photo["category"], photo["sha256"]))
    return inputs


def condition_photo(path, segment):
    from PIL import Image, ImageOps

    with Image.open(path) as original:
        photo = ImageOps.exif_transpose(original).convert("RGBA")
    photo.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
    alpha = photo.getchannel("A")
    if alpha.getextrema()[0] < 255:
        foreground = photo
        method = "source_alpha"
    else:
        foreground = segment(photo.convert("RGB")).convert("RGBA")
        method = "rembg_u2netp"
    alpha = foreground.getchannel("A")
    bounds = alpha.point(lambda value: 255 if value >= 16 else 0).getbbox()
    if bounds is None:
        if method == "source_alpha":
            raise ValueError("Input photo has no visible pixels")
        # Preserve a useful conditioning input when glass defeats segmentation.
        foreground = photo
        method = "original_rgb_empty_mask"
    else:
        foreground = foreground.crop(bounds)
    foreground = ImageOps.contain(foreground, (256, 256), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (INPUT_SIZE, INPUT_SIZE), GRAY)
    offset = ((INPUT_SIZE - foreground.width) // 2, (INPUT_SIZE - foreground.height) // 2)
    canvas.paste(foreground, offset, foreground.getchannel("A"))
    return canvas, method


def write_json(path, data):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def prepare_photos(output_dir):
    # Process rembg in a separate interpreter so its CPU memory is released before CUDA load.
    from rembg import new_session, remove

    report_path = output_dir / "run.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    session = None

    def segment(image):
        nonlocal session
        if session is None:
            session = new_session(MASK_MODEL, providers=["CPUExecutionProvider"])
        return remove(image, session=session)

    for index, record in enumerate(report["photos"], start=1):
        try:
            photo, method = condition_photo(Path(record["input"]), segment)
            filename = f"{index:02d}-{Path(record['input']).stem}-condition.png"
            photo.save(output_dir / filename)
            record.update(status="prepared", condition=filename, preprocessing=method)
            print(f"prepared {index}/{len(report['photos'])}: {record['input']}", flush=True)
        except Exception as exc:
            record.update(status="failed", stage="preprocessing", error=f"{type(exc).__name__}: {exc}")
            print(f"preprocessing failed: {record['input']}: {exc}", file=sys.stderr, flush=True)
        write_json(report_path, report)


def load_pipeline(torch, vae_tiling):
    from diffusers import DiffusionPipeline, EulerAncestralDiscreteScheduler
    from huggingface_hub import hf_hub_download

    custom_code = hf_hub_download(PIPELINE_ID, "pipeline.py", revision=PIPELINE_REVISION)
    pipe = DiffusionPipeline.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION, custom_pipeline=str(Path(custom_code).parent),
        torch_dtype=torch.float16, use_safetensors=True, low_cpu_mem_usage=True,
    )
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(
        pipe.scheduler.config, timestep_spacing="trailing",
    )
    if vae_tiling:
        pipe.enable_vae_tiling()
    return pipe.to("cuda")


def generate_grid(pipe, torch, photo, seed, steps):
    # Upstream condition latents and reference UNet noise also use global Torch RNG.
    torch.manual_seed(seed)
    generator = torch.Generator(device="cuda").manual_seed(seed)
    with torch.inference_mode():
        grid = pipe(photo, num_inference_steps=steps, guidance_scale=4.0, generator=generator).images[0]
    if grid.size != GRID_SIZE:
        raise ValueError(f"Expected grid {GRID_SIZE}, got {grid.size}")
    grid = grid.convert("RGB")
    if all(low == high for low, high in grid.getextrema()):
        raise ValueError("Zero123++ returned a uniform grid")
    return grid


def run(inputs, output_dir, seed, steps, vae_tiling, batch):
    from PIL import Image

    output_dir.mkdir(parents=True, exist_ok=False)
    report_path = output_dir / "run.json"
    report = dict(
        status="started", model=MODEL_ID, model_revision=MODEL_REVISION,
        pipeline=PIPELINE_ID, pipeline_revision=PIPELINE_REVISION, mask_model=MASK_MODEL,
        dataset_revision=DATASET_REVISION if batch else None, seed=seed, inference_steps=steps,
        guidance_scale=4.0, vae_tiling=vae_tiling, grid_size=list(GRID_SIZE),
        photos=[dict(input=str(path.resolve()), category=category, input_sha256=digest,
                     status="pending") for path, category, digest in inputs],
    )
    write_json(report_path, report)
    try:
        child_env = dict(os.environ, NUMBA_DISABLE_JIT="1", OMP_NUM_THREADS="2")
        try:
            subprocess.run([sys.executable, str(Path(__file__).resolve()), "--prepare-dir", str(output_dir)],
                           env=child_env, check=True)
        finally:
            # A failed subprocess may already have recorded preparation progress.
            report = json.loads(report_path.read_text(encoding="utf-8"))
        if not any(record["status"] == "prepared" for record in report["photos"]):
            raise RuntimeError("None of the input photographs could be prepared")
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable. Select a Colab GPU runtime or install CUDA PyTorch.")
        free, total = torch.cuda.mem_get_info()
        report["environment"] = dict(
            python=sys.version.split()[0], gpu=torch.cuda.get_device_name(0), cuda=torch.version.cuda,
            vram_free_bytes_at_start=free, vram_total_bytes=total,
            packages={name: importlib.metadata.version(name) for name in (
                "torch", "torchvision", "diffusers", "transformers", "huggingface-hub", "accelerate",
                "numpy", "Pillow", "rembg", "onnxruntime", "safetensors",
            )},
        )
        write_json(report_path, report)
        print(f"Loading {MODEL_ID} on {report['environment']['gpu']} (FP16)", flush=True)
        pipe = load_pipeline(torch, vae_tiling)
        for index, record in enumerate(report["photos"], start=1):
            if record["status"] != "prepared":
                continue
            print(f"generating {index}/{len(report['photos'])}: {record['input']}", flush=True)
            try:
                torch.cuda.reset_peak_memory_stats()
                start = time.perf_counter()
                with Image.open(output_dir / record["condition"]) as condition:
                    grid = generate_grid(pipe, torch, condition, seed, steps)
                torch.cuda.synchronize()
                name = f"{index:02d}-{Path(record['input']).stem}-grid.png"
                target = output_dir / name
                grid.save(target)
                record.update(status="succeeded", grid=name, grid_sha256=sha256(target),
                              seconds=round(time.perf_counter() - start, 3),
                              peak_vram_allocated_bytes=torch.cuda.max_memory_allocated())
            except Exception as exc:
                record.update(status="failed", stage="inference", error=f"{type(exc).__name__}: {exc}")
                print(f"inference failed: {record['input']}: {exc}", file=sys.stderr, flush=True)
                torch.cuda.empty_cache()
            write_json(report_path, report)
    except Exception as exc:
        report["fatal_error"] = f"{type(exc).__name__}: {exc}"
        report["status"] = "failed"
        write_json(report_path, report)
        raise
    succeeded = sum(record["status"] == "succeeded" for record in report["photos"])
    report.update(status="succeeded" if succeeded == len(inputs) else "failed", succeeded=succeeded)
    write_json(report_path, report)
    print(f"Saved {succeeded}/{len(inputs)} grids. Report: {report_path}", flush=True)
    return succeeded == len(inputs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--photo", type=Path)
    mode.add_argument("--batch", action="store_true")
    mode.add_argument("--prepare-dir", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--photos-dir", type=Path, default=Path("data/test_photos"))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=28)
    parser.add_argument("--vae-tiling", action="store_true")
    args = parser.parse_args()
    if args.prepare_dir:
        prepare_photos(args.prepare_dir)
        return 0
    if args.output_dir is None:
        parser.error("--output-dir is required")
    if not 0 <= args.seed < 2**63 or not 1 <= args.steps <= 1000:
        parser.error("Choose a seed in [0, 2**63) and steps in [1, 1000]")
    inputs = (load_batch(args.photos_dir) if args.batch
              else [(args.photo, None, sha256(args.photo))])
    return 0 if run(inputs, args.output_dir.resolve(), args.seed, args.steps, args.vae_tiling, args.batch) else 1


if __name__ == "__main__":
    sys.exit(main())
