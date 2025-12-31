#!/bin/bash
# ==============================================================================
# run_native.sh - Start inference-service with Metal GPU acceleration
# ==============================================================================
#
# This script runs the inference-service NATIVELY on macOS with Metal GPU
# acceleration. This is ~50-100x faster than Docker on Mac.
#
# When to use this script:
#   - Local development on Apple Silicon Mac
#   - Testing/debugging with fast inference
#   - Any time you need sub-second responses
#
# When to use Docker instead:
#   - CI/CD pipelines
#   - Linux server deployment
#   - CUDA GPU servers
#   - Reproducible builds
#
# Prerequisites:
#   - Python 3.11+ with venv
#   - llama-cpp-python installed with Metal support
#   - GGUF model files in ./models/
#
# Usage:
#   ./run_native.sh              # Start with D4 preset (default)
#   ./run_native.sh S1           # Start with specific preset
#   INFERENCE_PORT=9000 ./run_native.sh  # Custom port
#
# ==============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}=============================================${NC}"
echo -e "${BLUE}  inference-service - Native Metal Mode${NC}"
echo -e "${BLUE}=============================================${NC}"

# Check for Apple Silicon
if [[ "$(uname -m)" != "arm64" ]]; then
    echo -e "${YELLOW}Warning: Not running on Apple Silicon. Metal acceleration may not work.${NC}"
fi

# Check/create venv
if [ ! -d ".venv" ]; then
    echo -e "${YELLOW}Creating virtual environment...${NC}"
    python3 -m venv .venv
fi

# Activate venv
source .venv/bin/activate

# Check for llama-cpp-python with Metal
if ! python -c "import llama_cpp" 2>/dev/null; then
    echo -e "${YELLOW}Installing llama-cpp-python with Metal support...${NC}"
    echo -e "${YELLOW}This may take 5-10 minutes to compile...${NC}"
    
    # Install with Metal support
    CMAKE_ARGS="-DGGML_METAL=on" pip install llama-cpp-python --force-reinstall --no-cache-dir
    
    if [ $? -ne 0 ]; then
        echo -e "${RED}Failed to install llama-cpp-python with Metal${NC}"
        echo -e "${YELLOW}Try: pip install llama-cpp-python (without Metal)${NC}"
        exit 1
    fi
fi

# Install other requirements
echo -e "${BLUE}Installing dependencies...${NC}"
pip install -e . -q

# Set environment variables for native Metal mode
export INFERENCE_GPU_LAYERS="-1"  # -1 = All layers on Metal GPU

# Models directory: check multiple locations
# Priority: 1) INFERENCE_MODELS_DIR env var, 2) ai-models repo, 3) local ./models
if [ -z "$INFERENCE_MODELS_DIR" ]; then
    if [ -d "/Users/kevintoles/POC/ai-models/models" ]; then
        export INFERENCE_MODELS_DIR="/Users/kevintoles/POC/ai-models/models"
    elif [ -d "${SCRIPT_DIR}/../ai-models/models" ]; then
        export INFERENCE_MODELS_DIR="${SCRIPT_DIR}/../ai-models/models"
    else
        export INFERENCE_MODELS_DIR="${SCRIPT_DIR}/models"
    fi
fi

export INFERENCE_CONFIG_DIR="${SCRIPT_DIR}/config"
export INFERENCE_PORT="${INFERENCE_PORT:-8085}"
export INFERENCE_HOST="${INFERENCE_HOST:-0.0.0.0}"
export INFERENCE_LOG_LEVEL="${INFERENCE_LOG_LEVEL:-INFO}"
export INFERENCE_ENVIRONMENT="${INFERENCE_ENVIRONMENT:-development}"

# Optional: Set default preset from command line arg
if [ -n "$1" ]; then
    export INFERENCE_DEFAULT_PRESET="$1"
else
    export INFERENCE_DEFAULT_PRESET="${INFERENCE_DEFAULT_PRESET:-D4}"
fi

echo ""
echo -e "${GREEN}Configuration:${NC}"
echo -e "  GPU Layers:    ${GREEN}-1 (Metal - ALL LAYERS)${NC}"
echo -e "  Models Dir:    $INFERENCE_MODELS_DIR"
echo -e "  Config Dir:    $INFERENCE_CONFIG_DIR"
echo -e "  Port:          $INFERENCE_PORT"
echo -e "  Default Preset: $INFERENCE_DEFAULT_PRESET"
echo ""
echo -e "${GREEN}Starting inference-service with Metal acceleration...${NC}"
echo -e "${YELLOW}Press Ctrl+C to stop${NC}"
echo ""

# Start the service
python -m uvicorn src.main:app \
    --host "$INFERENCE_HOST" \
    --port "$INFERENCE_PORT" \
    --reload \
    --reload-dir src
