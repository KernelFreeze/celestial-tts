# syntax=docker/dockerfile:1
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

WORKDIR /app

# Install system dependencies for audio processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsox-dev \
    sox \
    && rm -rf /var/lib/apt/lists/*

# Enable bytecode compilation and link mode for faster startup
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

# Install dependencies first (cached layer)
COPY pyproject.toml uv.lock README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --no-dev --extra runpod

# Copy source code
COPY . .

# Install the project itself
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --extra runpod

# Pre-download Qwen3-TTS models to eliminate cold start
# This downloads the model weights during build time
ENV HF_HOME=/app/.cache/huggingface
ENV TRANSFORMERS_CACHE=/app/.cache/huggingface
RUN --mount=type=cache,target=/root/.cache/huggingface \
    uv run python scripts/download_models.py

EXPOSE 8080

RUN chmod +x /app/scripts/healthcheck.sh
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD /app/scripts/healthcheck.sh

CMD if [ -n "$RUNPOD_POD_ID" ]; then \
    uv run python runpod_handler.py; \
    else \
    uv run celestial-tts --host 0.0.0.0; \
    fi
