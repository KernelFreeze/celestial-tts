#!/bin/sh
# Health check script for Celestial TTS
# In serverless mode (RUNPOD_POD_ID set), always returns healthy
# In HTTP mode, checks the FastAPI health endpoint

if [ -n "$RUNPOD_POD_ID" ]; then
    # Serverless mode - RunPod handles health checking via the handler
    exit 0
else
    # HTTP mode - check the FastAPI health endpoint
    python -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/api/health')" 2>/dev/null || exit 1
fi
