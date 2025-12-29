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
│   ├── providers/          # LLM providers (llamacpp, vllm)
│   ├── orchestration/      # Multi-model orchestration
│   └── services/           # Model manager, queue manager
├── tests/
│   ├── unit/               # Unit tests
│   └── integration/        # Integration tests
├── config/                 # Configuration presets
├── docker/                 # Docker files
└── docs/                   # Documentation
```

## Documentation

- [Architecture](docs/ARCHITECTURE.md) - Full system design
- [API Reference](docs/API.md) - OpenAPI documentation
- [WBS](docs/WBS.md) - Work breakdown structure

## License

MIT License - see [LICENSE](LICENSE) for details.
