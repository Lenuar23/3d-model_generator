FROM ghcr.io/astral-sh/uv:0.12.17 AS uv
FROM python:3.12-slim AS web

COPY --from=uv /uv /usr/local/bin/uv
WORKDIR /app
ENV PYTHONUNBUFFERED=1 GRADIO_SERVER_NAME=0.0.0.0
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --locked --extra ui --no-install-project
COPY services/web ./services/web
EXPOSE 7860
CMD ["/app/.venv/bin/python", "services/web/app.py"]

FROM nvidia/cuda:12.1.1-devel-ubuntu22.04 AS generator

COPY --from=uv /uv /usr/local/bin/uv
WORKDIR /app
ENV UV_PYTHON_INSTALL_DIR=/opt/python \
    PYTHONUNBUFFERED=1
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN uv venv --python 3.12 .venv
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
ENV PYTHONPATH=/opt/hunyuan HF_HOME=/app/data/hunyuan_cache U2NET_HOME=/app/data/u2net \
    HUNYUAN_PAINT_DOWNLOAD=1
RUN /app/.venv/bin/python -c "import torch, custom_rasterizer; from hy3dgen.texgen.pipelines import Hunyuan3DPaintPipeline"
COPY services/generator ./services/generator
CMD ["/app/.venv/bin/python", "services/generator/server.py"]

FROM python:3.13-slim AS fast
COPY --from=uv /uv /usr/local/bin/uv
WORKDIR /app
ENV PYTHONUNBUFFERED=1
RUN apt-get update && apt-get install -y --no-install-recommends git libgl1 libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*
COPY services/fast/requirements-*.txt ./
RUN uv venv .venv-recognition && uv pip install --python .venv-recognition/bin/python \
        --index-strategy unsafe-best-match -r requirements-recognition.txt \
    && mkdir .venv-recognition/Scripts \
    && ln -s ../bin/python .venv-recognition/Scripts/python.exe
RUN uv venv .venv-wonder3d && uv pip install --python .venv-wonder3d/bin/python \
        --index-strategy unsafe-best-match -r requirements-wonder3d.txt \
    && mkdir .venv-wonder3d/Scripts \
    && ln -s ../bin/python .venv-wonder3d/Scripts/python.exe
RUN git clone https://github.com/xxlong0/Wonder3D.git /opt/wonder-source \
    && git -C /opt/wonder-source checkout --detach d894f827aa8c2917761a0dad3ab40df74c7a5b24
RUN uv pip install --system numpy==2.5.4 opencv-python-headless==4.14.0.94 \
    pillow==11.3.0 scipy==1.18.1 trimesh==4.12.2 plotly==6.9.0 scikit-image==0.26.0 \
    huggingface-hub==1.33.0 hf-xet==1.8.0
COPY services/generator/server.py ./server.py
COPY services/fast ./
RUN python -c "from photo2mesh.wonder3d_adapter import reconstruct_wonder"
CMD ["python", "fast_server.py"]
