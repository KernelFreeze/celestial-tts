# syntax=docker/dockerfile:1
FROM nvidia/cuda:12.8.1-runtime-ubuntu24.04 AS builder

# Get uv from its official image
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

WORKDIR /app

ENV TZ=Etc/UTC
ENV DEBIAN_FRONTEND=noninteractive

# Install Python 3.12 and system dependencies for audio processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.12 \
    python3.12-dev \
    libsox-dev \
    sox \
    && rm -rf /var/lib/apt/lists/*

# Point uv at the system Python
ENV UV_PYTHON=/usr/bin/python3.12

# Prevent a 3 hour compilation of flash-attn
ENV FLASH_ATTENTION_SKIP_CUDA_BUILD=TRUE

# Enable bytecode compilation and link mode for faster startup
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

# Install dependencies first (cached layer)
COPY pyproject.toml uv.lock README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --no-dev --extra runpod --extra cuda --preview-features extra-build-dependencies

# Copy source code
COPY . .

# Install the project itself
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --extra runpod --extra cuda --preview-features extra-build-dependencies

FROM nvidia/cuda:12.8.1-runtime-ubuntu24.04

WORKDIR /app

ENV TZ=Etc/UTC
ENV DEBIAN_FRONTEND=noninteractive

# Install only runtime dependencies (no -dev packages, no uv)
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.12 \
    sox \
    && rm -rf /var/lib/apt/lists/*

# Copy the pre-built venv from the builder stage
COPY --from=builder /app/.venv /app/.venv

# Copy source code (needed for runpod_handler.py and entry point modules)
COPY . .

# Use the venv directly via PATH instead of uv run
ENV PATH="/app/.venv/bin:$PATH"

EXPOSE 8080

RUN chmod +x /app/scripts/healthcheck.sh
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD /app/scripts/healthcheck.sh

CMD if [ -n "$RUNPOD_POD_ID" ]; then \
    python runpod_handler.py; \
    else \
    celestial-tts --host 0.0.0.0; \
    fi
