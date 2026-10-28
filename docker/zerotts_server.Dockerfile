# zerotts_server (the ZeroTTS upstream, ADR 0005): onnxruntime on CPU, no GPU.
# Weights and config are not baked in: compose.zerotts.yaml mounts configs/ and models/ read-only
# at /srv, the repo layout, so configs/zerotts_server.toml's relative model_dir works unchanged.
# Build context is the repo root (its .dockerignore applies):
#   docker build -f docker/zerotts_server.Dockerfile -t zerotts-server .
FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.12.9 /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never

WORKDIR /app
# Dependencies first so source edits do not reinstall them. The `server` extra pins zerotts and
# onnxruntime (CPU build).
COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-dev --extra server --no-install-project
COPY README.md ./
COPY src ./src
RUN uv sync --locked --no-dev --extra server --no-editable

RUN useradd --system --no-create-home --uid 10001 zerotts
USER zerotts

ENV PATH="/app/.venv/bin:$PATH" \
    ZEROTTS_SERVER_CONFIG=/srv/configs/zerotts_server.toml \
    ZEROTTS_SERVER_HOST=0.0.0.0
EXPOSE 8080
# /health is 200 only after the model is loaded and warmed up. Port 8080 as in
# configs/zerotts_server.toml; start period covers loading on a slow node.
HEALTHCHECK --interval=10s --timeout=5s --start-period=120s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8080/health', timeout=3)"]
CMD ["zerotts-server"]
