# Gateway only; audio.cpp is a separate image (docker/audiocpp.Dockerfile) or runs elsewhere.
# Config and lexicon are mounted at /config (compose.yaml).
FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.12.9 /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never

WORKDIR /app
# Dependencies first so source edits do not reinstall them.
COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-dev --no-install-project
COPY README.md ./
COPY src ./src
RUN uv sync --locked --no-dev --no-editable

RUN useradd --system --no-create-home --uid 10001 gateway
USER gateway

ENV PATH="/app/.venv/bin:$PATH" \
    OMNIVOICE_GATEWAY_CONFIG=/config/gateway.toml \
    OMNIVOICE_GATEWAY_HOST=0.0.0.0
EXPOSE 8000
# Liveness only (/health); upstream reachability is /ready. Port 8000 as in configs/gateway.toml.
HEALTHCHECK --interval=10s --timeout=5s --start-period=10s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)"]
CMD ["omnivoice-gateway"]
