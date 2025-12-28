# Inference Service Architecture

> **Version:** 1.0.0  
> **Last Updated:** 2025-12-27  
> **Status:** Design Phase

## Table of Contents

1. [Overview](#overview)
2. [Kitchen Brigade Positioning](#kitchen-brigade-positioning)
3. [Folder Structure](#folder-structure)
4. [API Contract](#api-contract)
5. [Model Configuration](#model-configuration)
6. [Orchestration Modes](#orchestration-modes)
7. [Model Role Mapping](#model-role-mapping)
8. [Concurrency & Scaling](#concurrency--scaling)
9. [Configuration Reference](#configuration-reference)
10. [Health Checks](#health-checks)
11. [llm-gateway Integration](#llm-gateway-integration)

---

## Overview

The **inference-service** is a self-hosted LLM inference worker that runs local GGUF models using `llama-cpp-python` (Mac/Metal) with future support for `vLLM` (server/CUDA).

### Key Characteristics

| Aspect | Value |
|--------|-------|
| **Port** | 8085 |
| **Role** | Sous Chef Worker (Kitchen Brigade) |
| **Access** | Internal only (called by llm-gateway) |
| **API** | OpenAI-compatible |
| **Backend (Current)** | llama-cpp-python + Metal |
| **Backend (Future)** | vLLM + CUDA |

### Hardware Targets

| Environment | Hardware | Capacity |
|-------------|----------|----------|
| **Dev (Current)** | Mac M1 Pro 16GB | 1-2 concurrent, 1-3 models |
| **Server (Future)** | RTX 6000 Ada (48GB) + RTX 4090 (24GB) + 128GB RAM | 50-100+ concurrent, all models |

---

## Kitchen Brigade Positioning

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Kitchen Brigade                               │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌──────────────┐                                                   │
│  │   Customer   │ (External Apps)                                   │
│  └──────┬───────┘                                                   │
│         │                                                            │
│         ▼                                                            │
│  ┌──────────────┐     ┌─────────────────┐                          │
│  │    Router    │────▶│ External LLMs   │ (Anthropic, OpenAI)      │
│  │  (llm-gw)    │     └─────────────────┘                          │
│  │    :8080     │                                                   │
│  └──────┬───────┘     ┌─────────────────┐                          │
│         │             │ inference-svc   │ ◀── NEW                  │
│         └────────────▶│     :8085       │                          │
│                       │  (Local LLMs)   │                          │
│                       └─────────────────┘                          │
│                                                                      │
│  Other Services:                                                    │
│  - Expeditor (ai-agents) :8082                                      │
│  - Cookbook (semantic-search) :8081                                 │
│  - Sous Chef (code-orchestrator) :8083                              │
│  - Auditor (audit-service) :8084                                    │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Folder Structure

### inference-service Repository

```
inference-service/
├── .github/
│   └── workflows/
│       ├── ci.yml
│       └── release.yml
├── docs/
│   ├── ARCHITECTURE.md          # This document
│   └── API.md                   # OpenAPI documentation
├── src/
│   ├── __init__.py
│   ├── main.py                  # FastAPI app entry point
│   ├── api/
│   │   ├── __init__.py
│   │   ├── routes/
│   │   │   ├── __init__.py
│   │   │   ├── completions.py   # /v1/chat/completions
│   │   │   ├── models.py        # /v1/models, /load, /unload
│   │   │   └── health.py        # /health, /health/ready
│   │   └── dependencies.py      # FastAPI dependencies
│   ├── core/
│   │   ├── __init__.py
│   │   ├── config.py            # Pydantic Settings
│   │   └── logging.py           # Structured logging
│   ├── models/
│   │   ├── __init__.py
│   │   ├── requests.py          # ChatCompletionRequest
│   │   └── responses.py         # ChatCompletionResponse, Chunk
│   ├── providers/
│   │   ├── __init__.py
│   │   ├── base.py              # InferenceProvider ABC
│   │   ├── llamacpp.py          # llama-cpp-python provider
│   │   └── vllm.py              # vLLM provider (future)
│   ├── orchestration/
│   │   ├── __init__.py
│   │   ├── orchestrator.py      # Multi-model orchestration
│   │   ├── modes/
│   │   │   ├── __init__.py
│   │   │   ├── single.py        # Single model mode
│   │   │   ├── critique.py      # Generate → Critique → Revise
│   │   │   ├── debate.py        # Parallel generate → Reconcile
│   │   │   ├── ensemble.py      # All generate → Consensus
│   │   │   └── pipeline.py      # Draft → Refine → Validate
│   │   └── roles.py             # Model role assignments
│   └── services/
│       ├── __init__.py
│       ├── model_manager.py     # Load/unload models
│       └── queue_manager.py     # Request queue handling
├── tests/
│   ├── __init__.py
│   ├── conftest.py
│   ├── unit/
│   │   ├── providers/
│   │   ├── orchestration/
│   │   └── services/
│   └── integration/
├── config/
│   └── model_configs.yaml       # 33 configuration presets
├── docker/
│   ├── Dockerfile
│   ├── Dockerfile.cuda          # For server with vLLM
│   └── docker-compose.yml
├── .env.example
├── .gitignore
├── pyproject.toml
├── requirements.txt
└── README.md
```

### ai-models Repository

```
ai-models/
├── .gitignore                   # Ignores *.gguf, *.safetensors, etc.
├── README.md
├── config/
│   ├── models.yaml              # Model registry (metadata)
│   └── configs.yaml             # 33 configuration presets
├── scripts/
│   ├── download_models.py       # Hugging Face download helper
│   └── verify_models.py         # SHA256 verification
└── models/                      # GITIGNORED - actual model files
    ├── phi-4/
    │   └── phi-4-Q4_K_S.gguf
    ├── deepseek-r1-7b/
    │   └── DeepSeek-R1-Distill-Qwen-7B-Q4_K_M.gguf
    ├── qwen2.5-7b/
    │   └── qwen2.5-7b-instruct-q4_k_m.gguf
    ├── llama-3.2-3b/
    │   └── Llama-3.2-3B-Instruct-Q4_K_M.gguf
    └── phi-3-medium-128k/
        └── Phi-3-medium-128k-instruct-Q4_K_M.gguf
```

---

## API Contract

### Endpoints

| Method | Path | Purpose |
|--------|------|---------|
| POST | `/v1/chat/completions` | OpenAI-compatible chat completion |
| GET | `/v1/models` | List available/loaded models |
| POST | `/v1/models/{model_id}/load` | Load model into memory |
| POST | `/v1/models/{model_id}/unload` | Unload model from memory |
| GET | `/health` | Liveness check |
| GET | `/health/ready` | Readiness (has loaded models) |

### Chat Completion Request

```json
POST /v1/chat/completions
{
  "model": "phi-4",
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Explain async/await in Python."}
  ],
  "stream": false,
  "max_tokens": 1024,
  "temperature": 0.7,
  "stop": ["\n\n"],
  "task_type": "code",
  "orchestration_mode": "critique",
  "priority": 2
}
```

### Chat Completion Response (Non-Streaming)

```json
{
  "id": "chatcmpl-abc123",
  "object": "chat.completion",
  "created": 1703704800,
  "model": "phi-4",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "Async/await in Python..."
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 25,
    "completion_tokens": 150,
    "total_tokens": 175
  },
  "orchestration": {
    "mode": "critique",
    "models_used": ["phi-4", "deepseek-r1-7b"],
    "rounds": 2,
    "confidence": 0.92
  }
}
```

### Chat Completion Response (Streaming)

```
data: {"id":"chatcmpl-abc123","choices":[{"delta":{"role":"assistant"},"index":0}]}

data: {"id":"chatcmpl-abc123","choices":[{"delta":{"content":"Async"},"index":0}]}

data: {"id":"chatcmpl-abc123","choices":[{"delta":{"content":"/await"},"index":0}]}

data: {"id":"chatcmpl-abc123","choices":[{"delta":{},"finish_reason":"stop","index":0}]}

data: [DONE]
```

### Models List Response

```json
GET /v1/models
{
  "data": [
    {
      "id": "phi-4",
      "status": "loaded",
      "memory_mb": 8400,
      "context_length": 16384,
      "roles": ["primary", "thinker", "coder"]
    },
    {
      "id": "deepseek-r1-7b",
      "status": "available",
      "memory_mb": 4700,
      "context_length": 32768,
      "roles": ["thinker"]
    }
  ],
  "config": "D3",
  "orchestration_mode": "debate"
}
```

---

## Model Configuration

### Available Models

| Model | File | Size | Context | Role |
|-------|------|------|---------|------|
| `phi-4` | phi-4-Q4_K_S.gguf | 8.4GB | 16K | primary, thinker, coder |
| `deepseek-r1-7b` | DeepSeek-R1-Distill-Qwen-7B-Q4_K_M.gguf | 4.7GB | 32K | thinker |
| `qwen2.5-7b` | qwen2.5-7b-instruct-q4_k_m.gguf | 4.5GB | 32K | coder, primary, fast |
| `llama-3.2-3b` | Llama-3.2-3B-Instruct-Q4_K_M.gguf | 2.0GB | 8K | fast |
| `phi-3-medium-128k` | Phi-3-medium-128k-instruct-Q4_K_M.gguf | 7.5GB | 128K | longctx, thinker |

### Configuration Presets (33 Total)

#### Single (S) - 1 Model

| Config | Model | Size | Mode |
|--------|-------|------|------|
| `S1` | phi-4 | 8.4GB | single |
| `S2` | deepseek-r1-7b | 4.7GB | single |
| `S3` | qwen2.5-7b | 4.5GB | single |
| `S4` | llama-3.2-3b | 2.0GB | single |
| `S5` | phi-3-medium-128k | 7.5GB | single |

#### Dual (D) - 2 Models

| Config | Models | Size | Mode | Roles |
|--------|--------|------|------|-------|
| `D1` | phi-4 + llama-3.2-3b | 10.4GB | pipeline | llama=drafter, phi-4=validator |
| `D2` | phi-4 + qwen2.5-7b | 12.9GB | critique | phi-4=gen, qwen=critic |
| `D3` | phi-4 + deepseek-r1-7b | 13.1GB | debate | Both generate, phi-4 reconciles |
| `D4` | deepseek-r1-7b + qwen2.5-7b | 9.2GB | critique | qwen=gen, deepseek=critic |
| `D5` | qwen2.5-7b + llama-3.2-3b | 6.5GB | pipeline | llama=draft, qwen=refine |
| `D6` | phi-3-medium-128k + llama-3.2-3b | 9.5GB | pipeline | llama=draft, phi-3=expand |
| `D7` | deepseek-r1-7b + llama-3.2-3b | 6.7GB | critique | llama=gen, deepseek=critic |
| `D8` | phi-4 + phi-3-medium-128k | 15.9GB | critique | phi-3=gen, phi-4=critic |
| `D9` | phi-3-medium-128k + qwen2.5-7b | 12.0GB | critique | phi-3=gen, qwen=critic |
| `D10` | phi-3-medium-128k + deepseek-r1-7b | 12.2GB | debate | Long context debate |

#### Triple (T) - 3 Models

| Config | Models | Size | Mode | Roles |
|--------|--------|------|------|-------|
| `T1` | phi-4 + qwen2.5-7b + llama-3.2-3b | 14.9GB | pipeline | llama→qwen→phi-4 |
| `T2` | phi-4 + deepseek-r1-7b + llama-3.2-3b | 15.1GB | ensemble | All vote, phi-4 synthesizes |
| `T3` | deepseek-r1-7b + qwen2.5-7b + llama-3.2-3b | 11.2GB | pipeline | llama→qwen→deepseek |
| `T4` | phi-4 + deepseek-r1-7b + qwen2.5-7b | 17.6GB | debate | Full reasoning debate |
| `T5` | phi-3-medium-128k + qwen2.5-7b + llama-3.2-3b | 14.0GB | pipeline | llama→qwen→phi-3 |
| `T6` | phi-4 + phi-3-medium-128k + llama-3.2-3b | 17.9GB | pipeline | llama→phi-4→phi-3 |
| `T7` | phi-4 + phi-3-medium-128k + qwen2.5-7b | 20.4GB | ensemble | General+Long+Code |
| `T8` | phi-4 + phi-3-medium-128k + deepseek-r1-7b | 20.6GB | debate | Reasoning trio |
| `T9` | phi-3-medium-128k + deepseek-r1-7b + qwen2.5-7b | 16.7GB | ensemble | Long+Think+Code |
| `T10` | phi-3-medium-128k + deepseek-r1-7b + llama-3.2-3b | 14.2GB | pipeline | llama→deepseek→phi-3 |

#### Quad (Q) - 4 Models

| Config | Models | Size | Mode |
|--------|--------|------|------|
| `Q1` | phi-4 + deepseek-r1-7b + qwen2.5-7b + llama-3.2-3b | 19.6GB | ensemble |
| `Q2` | phi-4 + phi-3-medium-128k + qwen2.5-7b + llama-3.2-3b | 22.4GB | pipeline |
| `Q3` | phi-4 + phi-3-medium-128k + deepseek-r1-7b + llama-3.2-3b | 22.6GB | ensemble |
| `Q4` | phi-4 + phi-3-medium-128k + deepseek-r1-7b + qwen2.5-7b | 25.1GB | debate |
| `Q5` | phi-3-medium-128k + deepseek-r1-7b + qwen2.5-7b + llama-3.2-3b | 18.7GB | ensemble |

#### Quint (P) - 5 Models (All)

| Config | Models | Size | Mode |
|--------|--------|------|------|
| `P1` | ALL 5 | 27.1GB | ensemble |
| `P2` | ALL 5 | 27.1GB | pipeline |
| `P3` | ALL 5 | 27.1GB | debate |

### Hardware Recommendations

| RAM Pressure | Mac 16GB Configs |
|--------------|------------------|
| **Light** (other apps) | S1, S2, S4, D5 |
| **Medium** (VS Code + service) | D1, D2, D4, D7 |
| **Full** (dedicated) | T1, T3 |

---

## Orchestration Modes

### Mode Definitions

| Mode | Description | Min Models | Flow |
|------|-------------|------------|------|
| `single` | One model, no critique | 1 | Request → Model → Response |
| `critique` | Generate then critique | 2 | Request → A(gen) → B(critique) → A(revise) → Response |
| `debate` | Parallel then reconcile | 2 | Request → [A,B](parallel) → Compare → Reconcile → Response |
| `ensemble` | All vote, synthesize | 2+ | Request → [All](parallel) → Consensus → Response |
| `pipeline` | Sequential stages | 2-3 | Request → Fast(draft) → Specialist(refine) → Primary(validate) → Response |

### Critique Mode

```
┌─────────┐     ┌─────────┐     ┌─────────┐     ┌─────────┐
│ Request │────▶│ Model A │────▶│ Model B │────▶│ Model A │────▶ Response
└─────────┘     │ Generate│     │ Critique│     │ Revise  │
                └─────────┘     └─────────┘     └─────────┘
                                     │
                                     ▼
                              "Issues found:
                               - Hallucination in line 3
                               - Missing edge case"
```

### Debate Mode

```
┌─────────┐     ┌─────────────────────────┐     ┌───────────┐
│ Request │────▶│ Model A    │   Model B  │────▶│ Reconcile │────▶ Response
└─────────┘     │ (parallel) │  (parallel)│     │ (Model A) │
                └─────────────────────────┘     └───────────┘
                              │
                              ▼
                       Compare outputs:
                       - Agreement: 85%
                       - Conflicts: 2 points
```

### Ensemble Mode

```
┌─────────┐     ┌─────────────────────────────────┐     ┌───────────┐
│ Request │────▶│ Model A │ Model B │ Model C │...│────▶│ Synthesis │────▶ Response
└─────────┘     │    ▼    │    ▼    │    ▼    │   │     └───────────┘
                │  Ans A  │  Ans B  │  Ans C  │   │           │
                └─────────────────────────────────┘           ▼
                                                        Consensus: 92%
                                                        Agreed points: 5
                                                        Flagged: 1
```

### Pipeline Mode

```
┌─────────┐     ┌─────────┐     ┌─────────┐     ┌─────────┐
│ Request │────▶│  Fast   │────▶│  Coder/ │────▶│ Primary │────▶ Response
└─────────┘     │ (Draft) │     │ Thinker │     │(Validate)│
                └─────────┘     │ (Refine)│     └─────────┘
                                └─────────┘
```

---

## Model Role Mapping

### Role Definitions

| Role | Description | Task Types |
|------|-------------|------------|
| `primary` | Default/general purpose | general, summarize, explain, chat |
| `fast` | Quick responses | simple, quick, classify, extract |
| `coder` | Code generation/review | code, debug, review, refactor, test |
| `thinker` | Complex reasoning | analyze, reason, compare, debate, plan |
| `longctx` | Long document processing | document, summarize_long, rag |

### Model → Role Assignment

| Model | Primary Role | Secondary Roles |
|-------|--------------|-----------------|
| `phi-4` | `primary` | `thinker`, `coder` |
| `llama-3.2-3b` | `fast` | `primary` |
| `qwen2.5-7b` | `coder` | `primary`, `fast` |
| `deepseek-r1-7b` | `thinker` | `primary` |
| `phi-3-medium-128k` | `longctx` | `primary`, `thinker` |

### Task Type → Role Routing

| Task Type | Preferred Role | Fallback |
|-----------|----------------|----------|
| `code`, `debug`, `refactor` | `coder` | `primary` |
| `analyze`, `reason`, `plan` | `thinker` | `primary` |
| `simple`, `classify`, `extract` | `fast` | `primary` |
| `document`, `summarize_long` | `longctx` | `thinker` |
| (default) | `primary` | any |

### Orchestration Role Assignments

| Role | In `critique` | In `debate` | In `pipeline` |
|------|---------------|-------------|---------------|
| `primary` | Generator | Participant + Reconciler | Validator |
| `fast` | - | - | Drafter |
| `coder` | Generator (code tasks) | Participant | Refiner (code) |
| `thinker` | Critic | Participant | Refiner (analysis) |
| `longctx` | Generator (long docs) | Participant | - |

---

## Concurrency & Scaling

### Current State (Mac M1 Pro 16GB)

```
┌─────────────────────────────────────────────────────────┐
│                   inference-service                      │
├─────────────────────────────────────────────────────────┤
│  Request Queue (asyncio.Queue)                          │
│  ┌─────────────────────────────────────────────────┐    │
│  │  req_1  │  req_2  │  req_3  │  ...  │  (max 10) │    │
│  └─────────────────────────────────────────────────┘    │
│                         │                                │
│                         ▼                                │
│  ┌─────────────────────────────────────────────────┐    │
│  │              Model Workers (1 per model)         │    │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐         │    │
│  │  │ phi-4   │  │ qwen2.5 │  │ llama3.2│         │    │
│  │  │(1 conc) │  │(1 conc) │  │(1 conc) │         │    │
│  │  └─────────┘  └─────────┘  └─────────┘         │    │
│  └─────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────┘
```

### Future State (Server)

```
┌─────────────────────────────────────────────────────────┐
│                   inference-service                      │
├─────────────────────────────────────────────────────────┤
│  Request Queue (asyncio.Queue)                          │
│  ┌─────────────────────────────────────────────────┐    │
│  │  req_1  │  req_2  │  ...  │  req_100  │ (max)   │    │
│  └─────────────────────────────────────────────────┘    │
│                         │                                │
│                         ▼                                │
│  ┌─────────────────────────────────────────────────┐    │
│  │         vLLM Engine (Continuous Batching)        │    │
│  │  ┌─────────────────────────────────────────┐    │    │
│  │  │ RTX 6000 Ada (48GB)  │  RTX 4090 (24GB) │    │    │
│  │  │  - phi-4             │  - qwen2.5-7b    │    │    │
│  │  │  - phi-3-medium-128k │  - llama-3.2-3b  │    │    │
│  │  │  - deepseek-r1-7b    │                  │    │    │
│  │  └─────────────────────────────────────────┘    │    │
│  └─────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────┘
```

---

## Configuration Reference

### Environment Variables

```env
# =============================================================================
# Core Settings
# =============================================================================
INFERENCE_PORT=8085
INFERENCE_HOST=0.0.0.0
INFERENCE_LOG_LEVEL=INFO

# =============================================================================
# Model Storage
# =============================================================================
INFERENCE_MODELS_DIR=/Users/kevintoles/POC/ai-models/models

# =============================================================================
# Model Configuration (pick one)
# =============================================================================
INFERENCE_CONFIG=D3                      # Use preset (S1-S5, D1-D10, T1-T10, Q1-Q5, P1-P3)
# INFERENCE_PRELOAD_MODELS=phi-4,qwen2.5-7b  # Or explicit list (overrides CONFIG)

# =============================================================================
# Orchestration
# =============================================================================
INFERENCE_ORCHESTRATION_MODE=critique    # single | critique | debate | ensemble | pipeline
INFERENCE_CRITIQUE_MAX_ROUNDS=2
INFERENCE_CRITIQUE_THRESHOLD=0.8
INFERENCE_DEBATE_MAX_ROUNDS=3
INFERENCE_DEBATE_RECONCILE_MODEL=primary
INFERENCE_ENSEMBLE_MIN_AGREEMENT=0.7
INFERENCE_ENSEMBLE_SYNTHESIS_MODEL=primary
INFERENCE_PIPELINE_STAGES=draft,refine,validate

# =============================================================================
# Request Handling
# =============================================================================
INFERENCE_MAX_CONCURRENT_REQUESTS=10
INFERENCE_REQUEST_TIMEOUT=120
INFERENCE_MAX_CONCURRENT_PER_MODEL=1     # Mac: 1, Server: 4+

# =============================================================================
# Queue Behavior
# =============================================================================
INFERENCE_QUEUE_STRATEGY=fifo            # fifo | priority
INFERENCE_REJECT_WHEN_FULL=true

# =============================================================================
# Priority Queue (when QUEUE_STRATEGY=priority)
# =============================================================================
INFERENCE_PRIORITY_ENABLED=false
INFERENCE_PRIORITY_LEVELS=3              # 1=low, 2=normal, 3=high
INFERENCE_DEFAULT_PRIORITY=2

# =============================================================================
# Auto-Routing (when multiple models loaded)
# =============================================================================
INFERENCE_AUTO_ROUTE_ENABLED=false
INFERENCE_AUTO_ROUTE_STRATEGY=least_busy # least_busy | round_robin | capability_match
INFERENCE_AUTO_ROUTE_FALLBACK=true

# =============================================================================
# Model Role Mapping (JSON)
# =============================================================================
INFERENCE_MODEL_ROLES='{"phi-4":["primary","thinker","coder"],"qwen2.5-7b":["coder","primary","fast"],"llama-3.2-3b":["fast"],"deepseek-r1-7b":["thinker"],"phi-3-medium-128k":["longctx","thinker"]}'

# =============================================================================
# Hardware Settings
# =============================================================================
INFERENCE_GPU_LAYERS=-1                  # -1 = all layers on GPU (Metal/CUDA)
INFERENCE_BACKEND=llamacpp               # llamacpp | vllm
```

---

## Health Checks

### Endpoints

| Endpoint | Purpose | Success | Failure |
|----------|---------|---------|---------|
| `GET /health` | Liveness | 200 if service up | 503 if down |
| `GET /health/ready` | Readiness | 200 if models loaded | 503 if no models |

### Responses

```json
GET /health
{"status": "ok"}

GET /health/ready
{
  "status": "ready",
  "config": "D3",
  "loaded_models": ["phi-4", "deepseek-r1-7b"],
  "orchestration_mode": "debate"
}

GET /health/ready (not ready)
HTTP 503
{
  "status": "not_ready",
  "reason": "Models still loading",
  "progress": "1/2"
}
```

### Docker Health Check

```yaml
healthcheck:
  test: ["CMD", "curl", "-f", "http://localhost:8085/health"]
  interval: 60s
  timeout: 30s
  retries: 3
  start_period: 120s  # Allow time for model loading
```

---

## llm-gateway Integration

### Router Configuration

Add to `llm-gateway/src/providers/router.py`:

```python
MODEL_PREFIXES = {
    # ... existing prefixes ...
    "inference:": "inference",
    "local:": "inference",
}
```

### Provider Registration

```python
# In provider initialization
if settings.inference_enabled:
    providers["inference"] = InferenceServiceProvider(
        base_url=settings.inference_service_url,  # http://localhost:8085
        timeout=120.0,
    )
```

### Request Flow

```
Client Request: model="inference:phi-4"
       │
       ▼
┌─────────────────┐
│   llm-gateway   │
│     :8080       │
└────────┬────────┘
         │ strips prefix, forwards to
         ▼
┌─────────────────┐
│inference-service│
│     :8085       │
│ model="phi-4"   │
└─────────────────┘
```

### Environment Variables (llm-gateway)

```env
LLM_GATEWAY_INFERENCE_ENABLED=true
LLM_GATEWAY_INFERENCE_SERVICE_URL=http://localhost:8085
```

---

## Appendix: Model Download Commands

```bash
# Ensure huggingface-cli is installed
pip install huggingface_hub

# Download all models to ~/POC/ai-models/models/
cd ~/POC/ai-models/models

# phi-4 (8.4GB)
mkdir -p phi-4 && huggingface-cli download microsoft/phi-4-gguf phi-4-Q4_K_S.gguf --local-dir phi-4

# deepseek-r1-7b (4.7GB)
mkdir -p deepseek-r1-7b && huggingface-cli download unsloth/DeepSeek-R1-Distill-Qwen-7B-GGUF DeepSeek-R1-Distill-Qwen-7B-Q4_K_M.gguf --local-dir deepseek-r1-7b

# qwen2.5-7b (4.5GB)
mkdir -p qwen2.5-7b && huggingface-cli download Qwen/Qwen2.5-7B-Instruct-GGUF qwen2.5-7b-instruct-q4_k_m.gguf --local-dir qwen2.5-7b

# llama-3.2-3b (2.0GB)
mkdir -p llama-3.2-3b && huggingface-cli download hugging-quants/Llama-3.2-3B-Instruct-Q4_K_M-GGUF llama-3.2-3b-instruct-q4_k_m.gguf --local-dir llama-3.2-3b

# phi-3-medium-128k (7.5GB)
mkdir -p phi-3-medium-128k && huggingface-cli download microsoft/Phi-3-medium-128k-instruct-gguf Phi-3-medium-128k-instruct-Q4_K_M.gguf --local-dir phi-3-medium-128k
```

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | 2025-12-27 | Initial architecture design |
