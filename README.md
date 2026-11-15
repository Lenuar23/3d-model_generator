# 3D Model Soft

Converts a photo of a single object into a GLB model using Hunyuan3D-2mini. Gradio displays the model and lets you download it. Enable Hunyuan Paint to generate UV textures.

## Running the project

Requires Docker Compose and an NVIDIA GPU with CUDA accessible from containers. On Windows, use Docker Desktop with Linux containers and WSL 2.

```powershell
docker compose up --build -d
```

Open the [web interface](http://127.0.0.1:7861). The first generation may download model weights. To change the port or disable Paint by default, set `WEB_PORT` or `HUNYUAN_PAINT_DEFAULT=0` in `.env`.

```powershell
docker compose logs -f web generator
docker compose down
```

## Project structure

- `services/web/` contains the interface, photo upload handling, and GLB delivery.
- `services/generator/` contains the internal HTTP service, shape generation, and Hunyuan Paint. It has GPU access and processes one request at a time.
- `data/` contains local model weights, caches, and test photos. Compose mounts only `hunyuan_cache`, `hunyuan_paint_weights`, and `u2net`.
- `experiments/texture/` contains a standalone prototype for texturing an existing model.
- `docs/` contains instructions and previous comparisons.

The browser connects to `web`, which sends the photo and settings to `generator` over the internal Compose network and receives a GLB file. Only the web interface port is published.

## Tests without a GPU

```powershell
uv run --extra ui python -m unittest discover -s tests -v
```

## Generation modes

- **Fast (use Math)** runs Photo2Mesh 2.4 AI-Assisted. CLIP identifies the object category, Wonder3D generates six RGB views and six normal maps, and Photo2Mesh builds a Visual Hull, refines the surface using normals, and bakes the texture into a GLB file. Wonder3D requires an NVIDIA GPU. Model weights are downloaded on the first request and cached in `data/photo2mesh_models`.
- **Quality (use 3D model)** uses Hunyuan3D and optional Hunyuan Paint in the GPU-enabled `generator` container.
