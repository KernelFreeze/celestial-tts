#!/bin/bash
# Test RunPod handler locally using Podman
# This simulates the RunPod environment to debug configuration issues

set -e

# Parse arguments
DRY_RUN=false
if [ "$1" = "--dry-run" ] || [ "$1" = "-d" ]; then
    DRY_RUN=true
    echo "=== DRY RUN MODE ==="
    echo "Will only debug config, not start handler"
    echo ""
fi

echo "=== Local RunPod Handler Test ==="
echo ""

# Check if running in the right directory
if [ ! -f "runpod_handler.py" ]; then
    echo "Error: Must run from the celestial-tts directory"
    echo "Usage: cd /path/to/celestial-tts && ./scripts/test_runpod_local.sh [--dry-run]"
    exit 1
fi

# Build the container image
echo "Building container image..."
podman build -t celestial-tts-runpod-test -f Containerfile .

echo ""
echo "=== Running container with RunPod environment simulation ==="
echo ""

# Determine command based on dry-run mode
if [ "$DRY_RUN" = true ]; then
    # Just debug config and exit
    CONTAINER_CMD="uv run python scripts/debug_config.py"
else
    # Run the actual handler
    CONTAINER_CMD="uv run python runpod_handler.py"
fi

# Run the container with RunPod-like environment
# - RUNPOD_POD_ID triggers handler mode
# - CELESTIAL_INTEGRATED_MODELS__DEVICE_MAP=cuda:0 forces GPU
# - CELESTIAL_LOGGING_LEVEL=DEBUG for verbose output
podman run -it --rm \
    --name celestial-runpod-test \
    --gpus all \
    -e RUNPOD_POD_ID="local-test-pod-123" \
    -e CELESTIAL_INTEGRATED_MODELS__DEVICE_MAP="cuda:0" \
    -e CELESTIAL_INTEGRATED_MODELS__MAX_LOADED_MODELS="1" \
    -e CELESTIAL_INTEGRATED_MODELS__ENABLED="true" \
    -e CELESTIAL_BOOTSTRAP_CREATE_TOKEN="true" \
    -e CELESTIAL_LOGGING_LEVEL="DEBUG" \
    -e HF_HOME="/app/.cache/huggingface" \
    -v "$(pwd)/.cache:/app/.cache:Z" \
    celestial-tts-runpod-test \
    $CONTAINER_CMD

echo ""
echo "=== Test complete ==="

if [ "$DRY_RUN" = true ]; then
    echo "Dry run completed - check config values above"
    echo ""
    echo "To run the actual handler, omit --dry-run:"
    echo "  ./scripts/test_runpod_local.sh"
else
    echo "If the container started successfully, check the logs above for:"
    echo "  1. 'Configuration loaded: device_map=cuda:0'"
    echo "  2. 'Loading model qwen3-tts-1.7b-preset on device: cuda:0'"
    echo ""
    echo "If it shows 'cpu' instead of 'cuda:0', the env vars aren't being read correctly."
    echo ""
    echo "To debug config without loading models, use:"
    echo "  ./scripts/test_runpod_local.sh --dry-run"
fi
