# Project guidance

## Current scope

3D Model Soft generates a GLB from a single photo with Hunyuan3D-2mini.
Communicate with the user in Ukrainian; write commit messages in English.

The project has moved away from Zero123++ and silhouette/space-carving
reconstruction. Do not restore the removed services or old architecture unless
explicitly requested. `project_plan.md` and `docs/reconstruction_comparison.md`
are historical, not the current implementation specification.

## Entry points and responsibilities

- `scripts/run_photo_to_3d.py`: Gradio UI, model subprocess, cached GLB preview/download.
- `scripts/generate_hunyuan.py`: background removal, shape generation, vertex coloring, export.
- `scripts/photo_vertex_colors.py`: approximate single-photo vertex coloring.
- `scripts/paint_multiview.py`: Hunyuan Paint using an existing mesh,
  geometry-conditioned views and a UV texture. The UI runs it after shape generation
  when the Paint checkbox is selected. Docker enables it by default.
- `scripts/local_texture.py` and `scripts/texture_bake.py`: earlier experimental
  caption/material-patch approach; retained for comparison, not the preferred Paint path.
- `src/3d_model_soft/ui/gradio_compat.py`: workaround for Model3D in pinned Gradio 6.31.0.
- `tests/`: offline subprocess-boundary and UV-baking tests.

The package name starts with a digit. Use `importlib.import_module("3d_model_soft...")`
when importing it from scripts; ordinary `from 3d_model_soft ...` is invalid Python.

## Environments

Use Python 3.12 and uv. Keep UI and GPU environments separate:

- `data/gradio_runtime/Scripts/python.exe`: existing local UI environment.
- `data/hunyuan_runtime/Scripts/python.exe`: existing local GPU/texture environment.
- `.venv`: project environment created by uv; do not assume it contains GPU dependencies.

`uv sync --locked --extra ui` installs UI dependencies, not the full Hunyuan runtime.
Do not run project `uv sync` against the GPU environment: its NumPy/model dependency
versions differ. Do not upgrade torch, diffusers, transformers or Gradio casually.
Read `docs/local_paint.md` before changing the GPU setup.

These data/ environments, source checkouts, weights and build helpers are local assets,
not included in a fresh clone. Verify paths before using the commands below.
The Docker image builds the rasterizer with Linux CUDA tools and downloads
Paint weights on demand into the mounted data directory.

## Local commands (PowerShell, repository root)

Start the existing UI environment with the configured model interpreter:

```powershell
$env:HUNYUAN_PYTHON = "$PWD/data/hunyuan_runtime/Scripts/python.exe"
$env:PYTHONPATH = "$PWD/data/hunyuan_source"
$env:HF_HOME = "$PWD/data/hunyuan_cache"
$env:U2NET_HOME = "$PWD/data/u2net"
$env:NUMBA_DISABLE_JIT = "1"
& .\data\gradio_runtime\Scripts\python.exe scripts/run_photo_to_3d.py
```

Default address: http://127.0.0.1:7860. Change `GRADIO_SERVER_PORT` if occupied;
do not terminate an unrelated process to free the port. Keep local launches on
127.0.0.1; public sharing requires a user request.

Paint an existing model (replace input paths; choose a NEW output directory):

```powershell
& .\data\hunyuan_runtime\Scripts\python.exe scripts/paint_multiview.py `
  path/to/photo.jpg path/to/model.glb data/new_paint_run `
  --view-size 512 --max-faces 60000
```

This changes only the output copy. The max-faces option simplifies that copy.
See `docs/local_paint.md` for the CUDA compiler, MSVC, rasterizer and weights.

## Validation

Run checks relevant to the changes:

```powershell
& .\data\gradio_runtime\Scripts\python.exe -m unittest discover -s tests -v
& .\data\hunyuan_runtime\Scripts\python.exe -m unittest discover -s tests -p test_texture_bake.py -v
uv lock --check
git diff --check
```

UV tests may skip in the UI environment; execute them in the GPU environment
with xatlas installed. Offline tests do not prove successful model inference.
For inference changes, distinguish unit checks, actual GPU runs and visual review.
Inspect saved views and GLB from several angles before claiming better texture quality.

Known Paint constraints:

- Tested on RTX 4050 Laptop 6 GB with model CPU offload, 512-pixel views,
  1024-pixel atlas and a 60,000-triangle output copy.
- 256-pixel views produced severe color artifacts and were removed as an option.
- Sequential CPU offload failed because Paint directly reads learned embeddings
  from meta tensors. Keep model CPU offload unless a replacement is verified.
- GPU runs are memory-intensive; run them sequentially. Preserve the local source
  patches described in `docs/local_paint.md` when refreshing the Hunyuan checkout.
- Hidden surfaces are inferred. Current outputs can still have lighting and stem-color
  artifacts; do not describe them as faithful recovery of unseen details.

## Changes, data and Git

Keep changes focused and preserve unrelated user edits. Inspect status/diffs before
editing or committing. Never discard work with reset/clean to make a task easier.
Do not delete cached weights, environments, photos or output models as code cleanup.

`data/`, virtual environments and generated GLBs are ignored. Do not force-add them
or commit API keys, caches, compiler binaries or third-party checkouts. Small comparison
images and measured reports belong under `docs/assets/`; explain camera, geometry and
rendering differences rather than presenting an uncontrolled comparison as a benchmark.

Keep README and docs consistent with actual entry points. Use new output directories
for experiments, and leave original input GLBs untouched. Report what was changed,
what was actually tested and any remaining limitations. Commit/push when requested;
do not merge into main or rewrite published history without authorization.
