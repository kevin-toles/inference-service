#!/bin/bash
# Run inference-service with libvips library path set

# Set library path for libvips (installed via brew)
export DYLD_FALLBACK_LIBRARY_PATH="/opt/homebrew/lib:${DYLD_FALLBACK_LIBRARY_PATH:-}"

# Activate venv and run
cd /Users/kevintoles/POC/inference-service
source .venv/bin/activate

exec python -m uvicorn src.main:app --host :: --port 8085 --timeout-keep-alive 30 --limit-concurrency 20 "$@"
