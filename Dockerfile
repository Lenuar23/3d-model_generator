FROM ghcr.io/astral-sh/uv:0.12.17 AS uv
FROM python:3.12-slim

COPY --from=uv /uv /usr/local/bin/uv
WORKDIR /app
ENV UV_PYTHON_DOWNLOADS=never \
    PYTHONUNBUFFERED=1 \
    GRADIO_SERVER_NAME=0.0.0.0

COPY pyproject.toml uv.lock README.md ./
RUN uv sync --locked --extra ui --no-install-project
COPY src ./src
COPY scripts ./scripts

EXPOSE 7860
CMD ["/app/.venv/bin/python", "scripts/run_gradio_demo.py"]
