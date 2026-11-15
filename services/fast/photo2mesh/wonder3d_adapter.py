"""Run Wonder3D as an isolated provider and reconstruct with the local hull."""
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
from tempfile import mkdtemp

import numpy as np
from scipy import ndimage
import cv2
from PIL import Image

from .automatic_ai import ROOT, worker
from .camera import Camera
from .hull_quality import agreement, rasterize, profile_prior
from .image_processing import load_photo
from .multiview import reconstruct_views
from .pipeline import Settings, make_mask
from .visual_hull import View, carve

CHECKPOINT = "flamehaze1115/wonder3d-v1.0@d6d2efc033a06a74d3761268de7295c97e6935d2"
NAMES = ("front", "front_right", "right", "back", "left", "front_left")
REQUIRED_FILES = (".venv-wonder3d/Scripts/python.exe", ".venv-recognition/Scripts/python.exe",
                  "models/clip/model.safetensors", "models/wonder3d/unet/diffusion_pytorch_model.bin",
                  "models/wonder-source/mvdiffusion")


def installed(root=None):
    """Report whether the local Wonder3D and CLIP environments and weights are present."""
    root = Path(root or ROOT)
    return all((root / path).exists() for path in REQUIRED_FILES)


def camera_rotations(matrices):
    """Map Photo2Mesh's front-aligned world into each upstream OpenGL camera."""
    rotations = np.asarray(matrices, dtype=float)[:, :, :3]
    return rotations @ rotations[0].T


def conditioning(path, destination):
    """Center a masked object on white at the upstream 192/256 crop ratio."""
    photo = load_photo(path)
    mask = make_mask(photo, Settings()).mask
    y, x = np.where(mask)
    image = Image.fromarray(photo.rgb)
    clean = Image.new("RGB", image.size, "white")
    clean.paste(image, mask=Image.fromarray(mask))
    crop = clean.crop((x.min(), y.min(), x.max()+1, y.max()+1))
    factor = 192/max(crop.size)
    crop = crop.resize(tuple(max(1,round(factor*side)) for side in crop.size), Image.Resampling.LANCZOS)
    square = Image.new("RGB", (256,256), "white")
    square.paste(crop, ((256-crop.width)//2,(256-crop.height)//2))
    square.save(destination)
    return photo, mask


def excluded_names(excluded):
    """Normalize operator exclusions given as view names or 1-based view numbers."""
    names = set()
    for item in excluded or ():
        if isinstance(item, str) and item in NAMES:
            names.add(item)
        elif isinstance(item, (int, np.integer)) or (isinstance(item, str) and item.isdigit()):
            index = int(item)
            if not 1 <= index <= len(NAMES):
                raise ValueError(f"Generated view number {index} is outside 1–{len(NAMES)}.")
            names.add(NAMES[index-1])
        else:
            raise ValueError(f"Unknown Wonder3D view: {item}.")
    return names


def load_pairs(generated):
    """Load each RGB image, normal map and upstream camera matrix as one indexed record.

    Pairing is by the worker's output index, which also indexes ``names`` and
    ``matrices`` in result.json. Records are never re-sorted or filtered here.
    """
    generated = Path(generated)
    metadata = json.loads((generated/"result.json").read_text(encoding="utf-8"))
    if tuple(metadata["names"]) != NAMES:
        raise ValueError("Unexpected Wonder3D view ordering.")
    if len(metadata["matrices"]) != len(NAMES):
        raise ValueError("Wonder3D camera metadata does not match the six generated views.")
    rotations = camera_rotations(metadata["matrices"])
    records = []
    for index, (name, rotation) in enumerate(zip(NAMES, rotations)):
        with Image.open(generated/f"rgb-{index}.png") as source:
            rgb = np.array(source.convert("RGB"))
        with Image.open(generated/f"normal-{index}.png") as source:
            normal = np.array(source.convert("RGB"))
        if rgb.shape != (256,256,3) or normal.shape != rgb.shape or rgb.std() < 4:
            raise ValueError(f"Invalid generated RGB/normal pair: {name}.")
        records.append({"index": index, "name": name, "rgb": rgb, "normal": normal, "rotation": rotation})
    return metadata, records


def _identity(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def generate_wonder(path, category=None, progress=None, folder=None):
    """Recognize the object and generate six paired Wonder3D views into a review folder."""
    if not path:
        raise ValueError("Upload one photograph first.")
    if folder is None:
        root = ROOT / "outputs"
        root.mkdir(exist_ok=True)
        folder = Path(mkdtemp(prefix="wonder-", dir=root))
    folder = Path(folder)
    (folder/"source.sha256").write_text(_identity(path), encoding="utf-8")
    conditioning(path, folder/"condition.png")
    if progress:
        progress(.04, "Recognizing with local CLIP")
    recognition = worker("recognize",[Path(path).resolve()],folder/"recognition")[0]
    recognition["category_used"] = category or recognition["category"]
    generated = folder/"generation"
    python = ROOT / ".venv-wonder3d/Scripts/python.exe"
    if not python.exists():
        raise ValueError("Wonder3D environment is missing. See docs/wonder3d_integration.md.")
    if progress:
        progress(.12,"Generating paired Wonder3D images and normals")
    try:
        completed = subprocess.run([str(python), str(Path(__file__).with_name("wonder_worker.py")),
                                    str(folder/"condition.png"), str(generated)],
                                   cwd=ROOT, capture_output=True, encoding="utf-8", errors="replace",
                                   timeout=900, env={**os.environ,"HF_HUB_OFFLINE":"1"})
    except subprocess.TimeoutExpired as error:
        raise ValueError("Wonder3D exceeded 15 minutes. Retry when GPU load is lower.") from error
    (folder/"worker.log").write_text(completed.stdout+completed.stderr, encoding="utf-8")
    if completed.returncode:
        raise ValueError(f"Wonder3D exited {completed.returncode}. See {folder/'worker.log'}.")
    return folder, recognition


def review_gallery(folder):
    """Return labeled RGB and normal previews with matching numbers for operator review."""
    _, records = load_pairs(Path(folder)/"generation")
    rgb = [(Image.fromarray(r["rgb"]), f"{r['index']+1} · {r['name']} · RGB") for r in records]
    normals = [(Image.fromarray(r["normal"]), f"{r['index']+1} · {r['name']} · normal") for r in records]
    return rgb, normals


def reconstruct_wonder(path, resolution=64, extent=1.8, strength=.65, *, reuse=None, excluded=(),
                       consensus=False, progress=None, strategy="Automatic", category=None,
                       output_folder=None, rank_only=False, weighted=False):
    """Produce or reuse paired views, carve the orthographic hull, then optionally refine it.

    ``reuse`` is a folder previously returned by :func:`generate_wonder`. ``excluded``
    lists view names or 1-based numbers that the operator rejected during review.
    """
    if not path:
        raise ValueError("Upload one photograph first.")
    skip = excluded_names(excluded)
    if reuse:
        folder = Path(reuse).resolve()
        identity = folder/"source.sha256"
        if not identity.is_file() or identity.read_text(encoding="utf-8") != _identity(path):
            raise ValueError("The uploaded image changed. Generate views again before reconstructing.")
        recognition = json.loads((folder/"recognition/result.json").read_text(encoding="utf-8"))[0]
        recognition["category_used"] = category or recognition["category"]
    else:
        folder, recognition = generate_wonder(path, category, progress, folder=output_folder)
    photo = load_photo(path)
    mask = make_mask(photo, Settings()).mask
    return reconstruct_pairs(photo, mask, folder/"generation", folder, resolution, extent, strength,
                             consensus, progress, recognition, strategy, skip, rank_only=rank_only, weighted=weighted)


def _mask_symmetry_stats(mask):
    """Measure bounding width, interior hole ratio, and upper-body left-right asymmetry."""
    ys, xs = np.where(mask)
    if not len(ys):
        return 0, 0.0, 0.0
    width = int(xs.max() - xs.min() + 1)
    hole_ratio = float((ndimage.binary_fill_holes(mask) & ~mask).sum()) / max(1, int(mask.sum()))
    y0, y1 = int(ys.min()), int(ys.max())
    ymid = int(y0 + 0.45 * (y1 - y0))
    if ymid <= y0:
        return width, hole_ratio, 0.0
    low_xs = xs[ys >= ymid]
    split = int(round(float(np.median(low_xs if len(low_xs) else xs))))
    up_mask = mask[y0:ymid]
    left_up = int(up_mask[:, :split].sum())
    right_up = int(up_mask[:, split:].sum())
    upper_asym = abs(left_up - right_up) / max(1, left_up + right_up)
    return width, hole_ratio, upper_asym


def rank_view_exclusions(views, extent, prior=None):
    """Rank omitted inferred views by agreement with the retained silhouettes."""
    if len(views) < 4:
        return []
    scores = []
    for omitted in (view for view in views if view.inferred):
        kept = [view for view in views if view is not omitted]
        try:
            hull = carve(kept, resolution=24, extent=extent, prior=prior)
        except ValueError:
            scores.append((omitted.name, -1.0))
            continue
        overlaps = [
            agreement(np.isfinite(rasterize(hull.vertices, hull.faces, view)), view.mask)
            for view in kept
        ]
        # Give the observed photograph half of the total weight.
        scores.append((omitted.name, float((overlaps[0] + np.mean(overlaps[1:])) / 2)))
    return [name for name, _ in sorted(scores, key=lambda item: item[1], reverse=True)]


def reconstruct_pairs(photo, mask, generated, folder, resolution, extent, strength,
                      consensus, progress, recognition, strategy="Automatic", excluded=frozenset(), *, rank_only=False, weighted=False):
    metadata, records = load_pairs(generated)
    y, x = np.where(mask)
    height = float(np.ptp(y) + 1)
    observed = View(photo.rgb, mask, Camera(0, 0, height, float((x.min() + x.max()) / 2),
                                            float((y.min() + y.max()) / 2)), "Original")
    category = recognition.get("category_used", recognition["category"])
    rounded = category in {"apple", "ball"}
    scale, front_cy = 192.0, 127.5
    candidates, warnings, sym_stats = [], [], []
    for record in records:
        index, name, rgb, normal, rotation = (record[k] for k in ("index", "name", "rgb", "normal", "rotation"))
        converted = load_photo(Image.fromarray(rgb))
        try:
            region = make_mask(converted, Settings(shadow_strength=0))
            if region.mask.mean() < .04 or region.mask.mean() > .78 or any(
                    "image edge" in warning for warning in region.warnings):
                raise ValueError("clipped or implausible foreground")
        except ValueError as error:
            warnings.append(f"Wonder3D view {name} excluded: {error}.")
            continue
        sym_stats.append(_mask_symmetry_stats(region.mask))
        # Fixed square framing uses one shared scale and vertical center from the generated front.
        if index == 0:
            scale = float(region.bbox[3])
            front_cy = float(127.5 + np.clip(region.bbox[1] + (region.bbox[3] - 1) / 2 - 127.5, -18, 18))
        if name in excluded:
            warnings.append(f"Wonder3D view {name} excluded by the operator.")
            continue
        candidates.append((name, rgb, normal, rotation, region))
    _, obs_hole, _ = _mask_symmetry_stats(mask)
    widths = [w for w, _, _ in sym_stats if w > 0]
    rotational_ok = (
        obs_hole <= 0.004
        and (not sym_stats or (max(h for _, h, _ in sym_stats) <= 0.004
                               and max(a for _, _, a in sym_stats) <= 0.10))
        and (len(widths) < 3 or min(widths) / max(widths) >= 0.80)
    )
    hybrid = strategy == "Hybrid prior" or (
        strategy == "Automatic" and category in {"apple", "ball", "bottle", "vase"} and rotational_ok
    )
    prior = profile_prior(mask, rounded) if hybrid else None
    if prior is not None:
        extent = min(extent, max(1.15, 2.45 * float(prior[1].max())))
    views, pairs, gallery = [observed], [], []
    for name, rgb, normal, rotation, region in candidates:
        azimuth = float(np.degrees(np.arctan2(rotation[2, 0], rotation[2, 2])))
        # Tightly lock vertical center across orthographic views to prevent vertical inter-view shear.
        elevation = float(np.degrees(np.arcsin(np.clip(rotation[2, 1], -1, 1))))
        bx, by, bw, bh = region.bbox
        cx = 127.5 + np.clip(bx + (bw - 1) / 2 - 127.5, -18, 18)
        cy = front_cy + np.clip(by + (bh - 1) / 2 - front_cy, -2, 2)
        camera = Camera(azimuth, elevation, float(scale), float(cx), float(cy),
                        rotation=tuple(map(tuple, rotation)))
        # Downweight soft generated detail while retaining that view's silhouette.
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        interior = cv2.erode(region.mask.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
        sharpness = float(np.var(cv2.Laplacian(gray, cv2.CV_32F)[interior])) if interior.any() else 0
        view = View(rgb, region.mask, camera, name, True, float(np.clip(sharpness / 45, .35, 1)))
        views.append(view)
        pairs.append((view, normal))
        gallery.append((Image.fromarray(normal), name))
    if len(views) < 3:
        raise ValueError(f"Too few Wonder3D views remain after review and checks ({len(views)-1}). "
                         f"Inspect {folder}.")
    if rank_only:
        return rank_view_exclusions(views, extent, prior)
    result = reconstruct_views(views, int(resolution), extent, output_root=folder,
                               preparation_warnings=warnings, texture_consensus=consensus, prior=prior,
                               normal_observations=pairs if strength else (), normal_strength=strength,
                               normal_basis=(1, 1, 1), weighted=weighted,
                               progress=(lambda frac, label: progress(.55 + .44 * frac, label)) if progress else None)
    report = result.report
    basis_used = list(report.get("normal_refinement", {}).get("basis_used", [1, 1, 1]))
    report.update({"recognition": recognition, "generation_model": CHECKPOINT, "generation": metadata,
                   "strategy": "Wonder3D hybrid profile" if hybrid else "Wonder3D feature-preserving Visual Hull",
                   "profile_prior_applied": bool(hybrid),
                   "effective_volume_extent": extent, "generated_views": len(NAMES),
                   "operator_excluded": sorted(excluded, key=NAMES.index),
                   "normal_basis_signs": basis_used,
                   "accepted_views": [view.name for view in views],
                   "texture_weights": {view.name: view.texture_weight for view in views},
                   "artifacts": str(folder), "generation_folder": str(generated),
                   "silhouette_iou": {view.name: agreement(np.isfinite(rasterize(
                        result.hull.vertices, result.hull.faces, view)), view.mask) for view in views}})
    (folder / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return result, [(Image.fromarray(view.rgb), view.name) for view in views], [
        (Image.fromarray(view.mask), view.name) for view in views], gallery
