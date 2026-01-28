# inference-service

Self-hosted LLM inference service with multi-model orchestration.

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-green.svg)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

**inference-service** is a local LLM inference worker that runs GGUF models using `llama-cpp-python` (Mac/Metal) with future support for `vLLM` (server/CUDA). It operates behind [llm-gateway](../llm-gateway) as part of the Kitchen Brigade architecture.

### Key Features

- **OpenAI-compatible API** - Drop-in replacement for `/v1/chat/completions`
- **Multi-model orchestration** - Single, Critique, Debate, Ensemble, Pipeline modes
- **Metal acceleration** - Native Apple Silicon support via llama-cpp-python
- **Context management** - Handoff state, trajectory injection, compression
- **Request queuing** - FIFO and priority-based queuing
- **41 configuration presets** - Pre-tuned model combinations (S1-S8, D1-D15, T1-T13, Q1-Q7, P1-P6)

### Kitchen Brigade Role

```
                    llm-gateway (:8080)
                          │
              ┌───────────┼───────────┐
              │           │           │
              ▼           ▼           ▼
        External     inference    Other
          LLMs       -service    Services
       (Anthropic)   (:8085)
```

**Role:** Sous Chef Worker  
**Port:** 8085  
**Access:** Internal only (via llm-gateway)

## Quick Start

### Prerequisites

- Python 3.11+
- Mac with Apple Silicon (for Metal acceleration) or Linux with CUDA
- GGUF model files (see [Model Configuration](#model-configuration))

### Installation

```bash
# Clone the repository
git clone https://github.com/kevintoles/inference-service.git
cd inference-service

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -e ".[dev]"

# Copy environment template
cp .env.example .env
# Edit .env with your model paths and settings
```

### Running the Service

#### Option 1: Native Mode (Mac - Recommended for Development)

**50-100x faster** than Docker on Mac - uses Metal GPU acceleration.

```bash
# Start with Metal acceleration (default D4 preset)
./run_native.sh

# Or with a specific preset
./run_native.sh S1
```

**Performance comparison:**

| Mode | Tokens/sec | "Hello" Response |
|------|------------|------------------|
| Native (Metal) | 30-100 tok/s | <1 second |
| Docker (CPU) | 0.04 tok/s | 3-4 minutes |

#### Option 2: Docker Mode (CI/CD, Linux Servers)

For deployment to Linux servers or CI/CD pipelines:

```bash
# Start with Docker
docker-compose -f docker/docker-compose.yml up -d

# Or with CUDA GPU support
docker-compose -f docker/docker-compose.yml --profile cuda up -d
```

> ⚠️ **Note:** Docker on macOS runs in a Linux VM which **cannot access Metal GPU**. Docker containers on Mac will use CPU-only inference (~50-100x slower). Use native mode for Mac development.

#### Option 3: Manual Uvicorn

```bash
# Start with default configuration
uvicorn src.main:app --host 0.0.0.0 --port 8085 --reload

# Or use the CLI entry point
inference-service
```

### Health Check

```bash
# Liveness check
curl http://localhost:8085/health

# Readiness check (returns loaded models)
curl http://localhost:8085/health/ready
```

## API Usage

### Chat Completions

```bash
curl -X POST http://localhost:8085/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "phi-4",
    "messages": [
      {"role": "user", "content": "Explain quantum computing"}
    ],
    "max_tokens": 500
  }'
```

### Streaming

```bash
curl -X POST http://localhost:8085/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "phi-4",
    "messages": [{"role": "user", "content": "Hello"}],
    "stream": true
  }'
```

### List Models

```bash
curl http://localhost:8085/v1/models
```

## Model Configuration

### Available Models

| Model | Context | Memory | Roles |
|-------|---------|--------|-------|
| llama-3.2-3b | 131K | ~2.5GB | fast |
| qwen2.5-7b | 131K | ~5GB | coder, primary |
| deepseek-r1-7b | 131K | ~5GB | thinker |
| granite-8b-code-128k | 128K | ~6GB | coder, longctx |
| phi-4 | 16K | ~9GB | primary, thinker, coder |
| phi-3-medium-128k | 128K | ~9GB | longctx, thinker |

### Configuration Presets

Set `INFERENCE_CONFIG` in `.env`:

| Preset | Models | Use Case |
|--------|--------|----------|
| S1-S8 | Single model | Simple inference |
| D1-D15 | Dual model | Critique, Debate |
| T1-T13 | Triple model | Pipeline, Ensemble |
| Q1-Q7 | Quad model | Complex orchestration |

## Orchestration Modes

| Mode | Flow | Best For |
|------|------|----------|
| `single` | One model generates | Simple queries |
| `critique` | Generate → Critique → Revise | Quality improvement |
| `debate` | Parallel → Reconcile | Verification |
| `ensemble` | All models → Consensus | Critical decisions |
| `pipeline` | Draft → Refine → Validate | Multi-step tasks |

## Development

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run only unit tests
pytest -m unit

# Run only integration tests (requires models)
pytest -m integration
```

### Code Quality

```bash
# Type checking
mypy src/

# Linting
ruff check src/

# Format
ruff format src/
```

### Project Structure

```
inference-service/
├── src/
│   ├── api/routes/         # FastAPI routes
│   ├── core/               # Config, logging, exceptions
│   ├── models/             # Pydantic request/response models
│   ├── providers/          # LLM providers (llamacpp, vllm, deepseek_vl)
│   ├── orchestration/      # Multi-model orchestration
│   └── services/           # Model manager, queue manager
├── tests/
│   ├── unit/               # Unit tests
│   └── integration/        # Integration tests
├── config/                 # Configuration presets
├── docker/                 # Docker files
└── docs/                   # Documentation
```

## Vision Language Model (VLM) Support

inference-service supports DeepSeek-VL2 for image classification and vision-language tasks.

### VLM Installation

```bash
# Install VLM dependencies
pip install -e ".[vlm]"

# Install DeepSeek-VL2 from GitHub (not on PyPI)
pip install --no-deps git+https://github.com/deepseek-ai/DeepSeek-VL2.git
```

### VLM Model Download

```bash
# Download DeepSeek-VL2-Tiny (~6.7GB)
cd /path/to/ai-models/models
git lfs install
git clone https://huggingface.co/deepseek-ai/deepseek-vl2-tiny
```

### VLM Configuration

Set these environment variables to enable VLM:

```bash
export INFERENCE_VISION_MODEL_PATH=deepseek-vl2-tiny
export INFERENCE_VISION_MODEL_ID=deepseek-vl2-tiny
export INFERENCE_MODELS_DIR=/path/to/ai-models/models
```

These are already configured in the platform startup scripts (`start_hybrid.sh`, `run_native.sh`).

### VLM API Usage

```bash
# Check VLM health
curl http://localhost:8085/api/v1/vision/health

# Classify an image
curl -X POST http://localhost:8085/api/v1/vision/classify \
  -H "Content-Type: application/json" \
  -d '{
    "image_path": "/path/to/image.png",
    "prompt": "Is this a technical diagram?",
    "max_tokens": 256
  }'
```

### VLM Notes

- **First request is slow** - Model loads on first use (~30-60 seconds for 6.7GB model)
- **MPS (Apple Silicon)** - Uses float32 for compatibility
- **Memory usage** - ~8-10GB RAM when loaded
- **transformers version** - Pinned to 4.38.x for DeepSeek-VL2 compatibility

## Documentation

- [Architecture](docs/ARCHITECTURE.md) - Full system design
- [API Reference](docs/API.md) - OpenAPI documentation
- [WBS](docs/WBS.md) - Work breakdown structure

## License

MIT License - see [LICENSE](LICENSE) for details.
