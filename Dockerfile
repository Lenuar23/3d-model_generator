FROM ghcr.io/astral-sh/uv:0.12.17 AS uv
FROM nvidia/cuda:12.1.1-devel-ubuntu22.04 AS web

COPY --from=uv /uv /usr/local/bin/uv
WORKDIR /app
ENV UV_PYTHON_INSTALL_DIR=/opt/python \
    PYTHONUNBUFFERED=1 \
    GRADIO_SERVER_NAME=0.0.0.0
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml uv.lock README.md ./
RUN uv sync --python 3.12 --locked --extra ui --no-install-project
COPY src ./src
COPY scripts ./scripts

EXPOSE 7860
CMD ["/app/.venv/bin/python", "scripts/run_photo_to_3d.py"]

FROM web AS photo-to-3d
RUN apt-get update && apt-get install -y --no-install-recommends git g++ libgl1 libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*
RUN uv pip install --no-cache --python .venv/bin/python torch==2.5.1 torchvision==0.20.1 \
    --index-url https://download.pytorch.org/whl/cu121
RUN uv pip install --no-cache --python .venv/bin/python \
    numpy==1.26.4 diffusers==0.32.2 transformers==4.48.3 accelerate==1.3.0 \
    huggingface-hub==0.28.1 einops omegaconf trimesh pymeshlab scikit-image \
    opencv-python==4.11.0.86 rembg==2.0.61 onnxruntime==1.20.1 \
    xatlas==0.0.11 pygltflib fast-simplification==0.2.0 ninja setuptools wheel
RUN git clone https://github.com/Tencent-Hunyuan/Hunyuan3D-2.git /opt/hunyuan \
    && git -C /opt/hunyuan checkout f8db63096c8282cb27354314d896feba5ba6ff8a
RUN CUDA_HOME=/usr/local/cuda TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;8.9+PTX" MAX_JOBS=2 \
    uv pip install --no-build-isolation --python .venv/bin/python /opt/hunyuan/hy3dgen/texgen/custom_rasterizer
COPY --from=web /app/.venv /app/.ui-venv
ENV PYTHONPATH=/opt/hunyuan HF_HOME=/app/data/hunyuan_cache U2NET_HOME=/app/data/u2net \
    HUNYUAN_PYTHON=/app/.venv/bin/python HUNYUAN_PAINT_DEFAULT=1 HUNYUAN_PAINT_DOWNLOAD=1
RUN /app/.venv/bin/python -c "import torch, custom_rasterizer; from hy3dgen.texgen.pipelines import Hunyuan3DPaintPipeline"
CMD ["/app/.ui-venv/bin/python", "scripts/run_photo_to_3d.py"]
