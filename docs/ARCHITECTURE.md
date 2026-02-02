# Inference Service Architecture

> **Version:** 1.5.0  
> **Last Updated:** 2026-01-27  
> **Status:** Production

## Table of Contents

1. [Overview](#overview)
2. [Kitchen Brigade Positioning](#kitchen-brigade-positioning)
3. [Folder Structure](#folder-structure)
4. [API Contract](#api-contract)
5. [Model Configuration](#model-configuration)
6. [Orchestration Modes](#orchestration-modes)
7. [Model Role Mapping](#model-role-mapping)
8. [Context Management](#context-management)
9. [Error Handling](#error-handling)
10. [Caching Strategy](#caching-strategy)
11. [Concurrency & Scaling](#concurrency--scaling)
12. [Configuration Reference](#configuration-reference)
13. [Health Checks](#health-checks)
14. [llm-gateway Integration](#llm-gateway-integration)
15. [Observability](#observability)

---

## Overview

The **inference-service** is a self-hosted LLM inference worker that runs local GGUF models using `llama-cpp-python` (Mac/Metal) with future support for `vLLM` (server/CUDA).

### Key Characteristics

| Aspect | Value |
|--------|-------|
| **Port** | 8085 |
| **Role** | Sous Chef Worker (Kitchen Brigade) |
| **Access** | Internal only (via CMS:8086 proxy or direct) |
| **API** | OpenAI-compatible |
| **Backend (Current)** | llama-cpp-python + Metal |
| **Backend (Future)** | vLLM + CUDA |
| **Observability** | OpenTelemetry tracing (OBS-11) |

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
│         │             │   CMS :8086     │ ◀── LLM Proxy            │
│         │             │ (Context Mgmt)  │                          │
│         │             └────────┬────────┘                          │
│         │                      │                                    │
│         │             ┌────────▼────────┐                          │
│         └────────────▶│ inference-svc   │                          │
│                       │     :8085       │                          │
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

### ai-models Repository (Storage Only)

> **⚠️ Configuration Ownership**: inference-service is the **single source of truth** for all model configuration. The ai-models repo is **storage-only** - it contains the actual .gguf files and download scripts, but NO configuration files.

```
ai-models/
├── .gitignore                   # Ignores *.gguf, *.safetensors, etc.
├── README.md
├── scripts/
│   └── download_models.py       # Hugging Face download helper (self-contained)
└── models/                      # GITIGNORED - actual model files
    ├── phi-4/
    │   └── phi-4-Q4_K_S.gguf
    ├── deepseek-r1-7b/
    │   └── DeepSeek-R1-Distill-Qwen-7B-Q4_K_M.gguf
    ├── qwen2.5-7b/
    │   └── qwen2.5-7b-instruct-q4_k_m.gguf
    ├── qwen3-8b/                                        # NEW
    │   └── Qwen3-8B-Q4_K_M.gguf
    ├── qwen3-coder-30b-a3b/                             # NEW (MoE)
    │   └── Qwen3-Coder-30B-A3B-Instruct-Q3_K_M.gguf
    ├── llama-3.2-3b/
    │   └── llama-3.2-3b-instruct-q4_k_m.gguf
    ├── phi-3-medium-128k/
    │   └── Phi-3-medium-128k-instruct-Q4_K_M.gguf
    ├── gpt-oss-20b/
    │   └── gpt-oss-20b-Q4_K_M.gguf
    ├── granite-8b-code-128k/
    │   └── granite-8b-code-instruct-128k.Q4_K_M.gguf
    └── granite-20b-code/
        └── granite-20b-code-instruct.Q4_K_M.gguf
```

### Configuration Ownership Summary

| Repo | Contains | Does NOT Contain |
|------|----------|------------------|
| **inference-service** | `models.yaml` (metadata, gpu_layers, context_length, roles), `presets.yaml` (35+ presets) | Model files |
| **ai-models** | `.gguf` model files, download scripts | Configuration files |

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

### Vision Endpoints (DeepSeek-VL)

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/v1/vision/health` | Vision model health check |
| POST | `/v1/vision/classify` | Single image classification |
| POST | `/v1/vision/classify/batch` | Batch image classification |
| POST | `/v1/vision/load` | Load vision model |
| POST | `/v1/vision/unload` | Unload vision model |

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

| Model | File | Size | Context | Role | Hardware |
|-------|------|------|---------|------|----------|
| `llama-3.2-3b` | llama-3.2-3b-instruct-q4_k_m.gguf | 2.0GB | 8K | fast | Mac ✅ |
| `qwen2.5-7b` | qwen2.5-7b-instruct-q4_k_m.gguf | 4.4GB | 32K | coder, primary | Mac ✅ |
| `deepseek-r1-7b` | DeepSeek-R1-Distill-Qwen-7B-Q4_K_M.gguf | 4.4GB | 32K | thinker | Mac ✅ |
| `granite-8b-code-128k` | granite-8b-code-instruct-128k.Q4_K_M.gguf | 4.5GB | 128K | coder, longctx | Mac ✅ |
| `phi-4` | phi-4-Q4_K_S.gguf | 7.9GB | 16K | primary, thinker, coder | Mac ✅ |
| `phi-3-medium-128k` | Phi-3-medium-128k-instruct-Q4_K_M.gguf | 8.6GB | 128K | longctx, thinker | Mac ✅ |
| `gpt-oss-20b` | gpt-oss-20b-Q4_K_M.gguf | 11.6GB | 32K | primary, thinker | Server 🖥️ |
| `granite-20b-code` | granite-20b-code-instruct.Q4_K_M.gguf | 12.8GB | 8K | coder | Server 🖥️ |

### Configuration Presets (41 Total)

#### Single (S) - 1 Model

| Config | Model | Size | Mode | Hardware |
|--------|-------|------|------|----------|
| `S1` | phi-4 | 7.9GB | single | Mac ✅ |
| `S2` | deepseek-r1-7b | 4.4GB | single | Mac ✅ |
| `S3` | qwen2.5-7b | 4.4GB | single | Mac ✅ |
| `S4` | llama-3.2-3b | 2.0GB | single | Mac ✅ |
| `S5` | phi-3-medium-128k | 8.6GB | single | Mac ✅ |
| `S6` | granite-8b-code-128k | 4.5GB | single | Mac ✅ |
| `S7` | gpt-oss-20b | 11.6GB | single | Server 🖥️ |
| `S8` | granite-20b-code | 12.8GB | single | Server 🖥️ |

#### Dual (D) - 2 Models

| Config | Models | Size | Mode | Roles | Hardware |
|--------|--------|------|------|-------|----------|
| `D1` | phi-4 + llama-3.2-3b | 9.9GB | pipeline | llama=drafter, phi-4=validator | Mac ✅ |
| `D2` | phi-4 + qwen2.5-7b | 12.3GB | critique | phi-4=gen, qwen=critic | Mac ⚠️ |
| `D3` | phi-4 + deepseek-r1-7b | 12.3GB | debate | Both generate, phi-4 reconciles | Mac ⚠️ |
| `D4` | deepseek-r1-7b + qwen2.5-7b | 8.8GB | critique | qwen=gen, deepseek=critic | Mac ✅ |
| `D5` | qwen2.5-7b + llama-3.2-3b | 6.4GB | pipeline | llama=draft, qwen=refine | Mac ✅ |
| `D6` | phi-3-medium-128k + llama-3.2-3b | 10.6GB | pipeline | llama=draft, phi-3=expand | Mac ✅ |
| `D7` | deepseek-r1-7b + llama-3.2-3b | 6.4GB | critique | llama=gen, deepseek=critic | Mac ✅ |
| `D8` | phi-4 + phi-3-medium-128k | 16.5GB | critique | phi-3=gen, phi-4=critic | Server 🖥️ |
| `D9` | phi-3-medium-128k + qwen2.5-7b | 13.0GB | critique | phi-3=gen, qwen=critic | Mac ⚠️ |
| `D10` | phi-3-medium-128k + deepseek-r1-7b | 13.0GB | debate | Long context debate | Mac ⚠️ |
| `D11` | granite-8b-code-128k + llama-3.2-3b | 6.5GB | pipeline | llama=draft, granite=code | Mac ✅ |
| `D12` | granite-8b-code-128k + deepseek-r1-7b | 8.9GB | critique | granite=gen, deepseek=critic | Mac ✅ |
| `D13` | granite-8b-code-128k + qwen2.5-7b | 8.9GB | debate | Code model debate | Mac ✅ |
| `D14` | granite-8b-code-128k + phi-4 | 12.4GB | critique | granite=gen, phi-4=critic | Mac ⚠️ |
| `D15` | gpt-oss-20b + granite-20b-code | 24.4GB | critique | gpt=gen, granite=critic | Server 🖥️ |

#### Triple (T) - 3 Models

| Config | Models | Size | Mode | Roles | Hardware |
|--------|--------|------|------|-------|----------|
| `T1` | phi-4 + qwen2.5-7b + llama-3.2-3b | 14.3GB | pipeline | llama→qwen→phi-4 | Mac ⚠️ |
| `T2` | phi-4 + deepseek-r1-7b + llama-3.2-3b | 14.3GB | ensemble | All vote, phi-4 synthesizes | Mac ⚠️ |
| `T3` | deepseek-r1-7b + qwen2.5-7b + llama-3.2-3b | 10.8GB | pipeline | llama→qwen→deepseek | Mac ✅ |
| `T4` | phi-4 + deepseek-r1-7b + qwen2.5-7b | 16.7GB | debate | Full reasoning debate | Server 🖥️ |
| `T5` | phi-3-medium-128k + qwen2.5-7b + llama-3.2-3b | 15.0GB | pipeline | llama→qwen→phi-3 | Server 🖥️ |
| `T6` | phi-4 + phi-3-medium-128k + llama-3.2-3b | 18.5GB | pipeline | llama→phi-4→phi-3 | Server 🖥️ |
| `T7` | phi-4 + phi-3-medium-128k + qwen2.5-7b | 20.9GB | ensemble | General+Long+Code | Server 🖥️ |
| `T8` | phi-4 + phi-3-medium-128k + deepseek-r1-7b | 20.9GB | debate | Reasoning trio | Server 🖥️ |
| `T9` | phi-3-medium-128k + deepseek-r1-7b + qwen2.5-7b | 17.4GB | ensemble | Long+Think+Code | Server 🖥️ |
| `T10` | phi-3-medium-128k + deepseek-r1-7b + llama-3.2-3b | 15.0GB | pipeline | llama→deepseek→phi-3 | Server 🖥️ |
| `T11` | granite-8b-code-128k + qwen2.5-7b + llama-3.2-3b | 10.9GB | pipeline | llama→qwen→granite (code) | Mac ✅ |
| `T12` | granite-8b-code-128k + deepseek-r1-7b + llama-3.2-3b | 10.9GB | pipeline | llama→deepseek→granite (think+code) | Mac ✅ |
| `T13` | granite-8b-code-128k + phi-4 + llama-3.2-3b | 14.4GB | pipeline | llama→phi-4→granite | Mac ⚠️ |

#### Quad (Q) - 4 Models

| Config | Models | Size | Mode | Hardware |
|--------|--------|------|------|----------|
| `Q1` | phi-4 + deepseek-r1-7b + qwen2.5-7b + llama-3.2-3b | 18.7GB | ensemble | Server 🖥️ |
| `Q2` | phi-4 + phi-3-medium-128k + qwen2.5-7b + llama-3.2-3b | 22.9GB | pipeline | Server 🖥️ |
| `Q3` | phi-4 + phi-3-medium-128k + deepseek-r1-7b + llama-3.2-3b | 22.9GB | ensemble | Server 🖥️ |
| `Q4` | phi-4 + phi-3-medium-128k + deepseek-r1-7b + qwen2.5-7b | 25.3GB | debate | Server 🖥️ |
| `Q5` | phi-3-medium-128k + deepseek-r1-7b + qwen2.5-7b + llama-3.2-3b | 19.4GB | ensemble | Server 🖥️ |
| `Q6` | granite-8b-code-128k + phi-4 + qwen2.5-7b + llama-3.2-3b | 18.8GB | pipeline | Server 🖥️ |
| `Q7` | granite-8b-code-128k + deepseek-r1-7b + qwen2.5-7b + llama-3.2-3b | 15.3GB | ensemble | Server 🖥️ |

#### Quint (P) - 5+ Models

| Config | Models | Size | Mode | Hardware |
|--------|--------|------|------|----------|
| `P1` | phi-4 + phi-3 + deepseek + qwen + llama | 27.3GB | ensemble | Server 🖥️ |
| `P2` | phi-4 + phi-3 + deepseek + qwen + llama | 27.3GB | pipeline | Server 🖥️ |
| `P3` | phi-4 + phi-3 + deepseek + qwen + llama | 27.3GB | debate | Server 🖥️ |
| `P4` | ALL 6 Mac models + gpt-oss-20b | 38.9GB | ensemble | Server 🖥️ |
| `P5` | ALL 6 Mac models + granite-20b-code | 40.1GB | pipeline | Server 🖥️ |
| `P6` | ALL 8 models | 53.2GB | ensemble | Server 🖥️ |

### Hardware Recommendations

| RAM Pressure | Mac 16GB Configs | Notes |
|--------------|------------------|-------|
| **Light** (<8GB) | S2, S3, S4, S6, D5, D7, D11 | Other apps running |
| **Medium** (8-12GB) | S1, S5, D1, D4, D6, D12, D13 | VS Code + service |
| **Full** (12-15GB) | D2, D3, D9, D10, D14, T1, T3 | Dedicated inference |
| **Server Only** | S7, S8, D8, D15, T4+, Q1+, P1+ | Requires 24GB+ VRAM |

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

| Model | Primary Role | Secondary Roles | Best For |
|-------|--------------|-----------------|----------|
| `llama-3.2-3b` | `fast` | `primary` | Quick drafts, simple queries |
| `qwen2.5-7b` | `coder` | `primary` | Code generation, technical tasks |
| `qwen3-8b` | `coder` | `primary` | Code generation, D4v2 preset with deepseek |
| `qwen3-coder-30b-a3b` | `coder` | `primary`, `thinker` | Standalone MoE code generation (3.3B active) |
| `deepseek-r1-7b` | `thinker` | `primary` | Chain-of-thought, complex reasoning |
| `granite-8b-code-128k` | `coder` | `longctx` | Full-file code analysis, 128K context |
| `phi-4` | `primary` | `thinker`, `coder` | General reasoning, summarization |
| `phi-3-medium-128k` | `longctx` | `thinker` | Long document processing |
| `gpt-oss-20b` | `primary` | `thinker` | High-capacity reasoning (server) |
| `granite-20b-code` | `coder` | `thinker` | Complex code tasks (server) |

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

## Context Management

### Core Principles

Context window management is critical for multi-model orchestration. Key insights:

| Principle | Description |
|-----------|-------------|
| **Context is a budget** | Finite, ephemeral, order-sensitive, prone to interference |
| **Errors are sticky** | Once incorrect assumptions enter context, they bias subsequent reasoning |
| **Compress conclusions, not process** | Preserve decisions, drop reasoning chains |
| **Control beats capacity** | More context ≠ better context |

### Context Window Constraints

| Model | Context | Usable Tokens | Generation Room |
|-------|---------|---------------|-----------------|
| llama-3.2-3b | 8K | ~6K after system | ~4K output |
| qwen2.5-7b | 32K | ~28K after system | ~16K output |
| qwen3-8b | 32K | ~28K after system | ~16K output |
| qwen3-coder-30b-a3b | 32K* | ~28K after system | ~16K output |
| deepseek-r1-7b | 32K | ~28K after system | ~16K output |
| granite-8b-code-128k | 128K | ~120K after system | ~64K output |
| phi-4 | 16K | ~14K after system | ~8K output |
| phi-3-medium-128k | 128K | ~120K after system | ~64K output |
| gpt-oss-20b | 32K | ~28K after system | ~16K output |
| granite-20b-code | 8K | ~6K after system | ~4K output |

> *Note: qwen3-coder-30b-a3b supports 256K native context (extendable to 1M), reduced to 32K for safe Mac 16GB operation.

### Context Budget Allocation

```python
MODEL_CONTEXT_BUDGETS = {
    "llama-3.2-3b": {
        "total": 8192,
        "system": 500,
        "trajectory": 300,
        "handoff": 1500,
        "user_query": 2000,
        "generation": 3892,
    },
    "granite-8b-code-128k": {
        "total": 131072,
        "system": 1000,
        "trajectory": 500,
        "handoff": 4000,
        "user_query": 32000,
        "generation": 93572,
    },
    # ... other models follow similar pattern
}
```

### Handoff State Object

For multi-model orchestration, context is passed via structured handoff objects, not raw text:

```python
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class HandoffState:
    """Structured state for model-to-model handoffs.
    
    Note: Uses field(default_factory=list) per CODING_PATTERNS_ANALYSIS AP-1.5
    to avoid mutable default argument anti-pattern.
    """
    request_id: str
    goal: str                    # Original user intent
    current_step: int            # Position in pipeline
    total_steps: int
    
    # Mutable fields use default_factory (AP-1.5 compliance)
    constraints: list[str] = field(default_factory=list)
    decisions_made: list[str] = field(default_factory=list)
    evidence_refs: list[str] = field(default_factory=list)
    active_errors: list[str] = field(default_factory=list)
    resolved_errors: list[str] = field(default_factory=list)
    
    # Optional fields explicitly annotated (AP-1.1 compliance)
    compressed_context: Optional[str] = None
```

### Trajectory Injection

Every prompt includes trajectory to prevent "lost agent" problem:

```
## Current Position
- Goal: {original_user_intent}
- Step: {current}/{total} - {step_name}
- Previous: {what_was_decided}
- Next: {what_this_model_must_decide}
- Forbidden: {what_not_to_do}
```

### Compression Strategy

Before model-to-model handoffs, context is compressed:

| Preserve | Drop |
|----------|------|
| Decisions made | Raw reasoning chains |
| Constraints discovered | Verbose explanations |
| Open questions | Stack traces |
| Key evidence refs | Intermediate outputs |
| Error summaries | Resolved error details |

### Automated Token Budget Enforcement

```python
async def fit_to_budget(
    content: str,
    max_tokens: int,
    model: Llama,
    max_iterations: int = 3
) -> str:
    """Iteratively compress until under budget.
    
    Note: Uses iterative approach per CODING_PATTERNS_ANALYSIS AP-2.1
    to avoid high cognitive complexity from recursion.
    """
    for iteration in range(max_iterations):
        current_tokens = _count_tokens(content, model)
        
        if current_tokens <= max_tokens:
            return content
        
        # Calculate compression ratio needed
        ratio = _calculate_compression_ratio(max_tokens, current_tokens)
        
        # Compress using fast model
        content = await _apply_compression(
            content=content,
            target_ratio=ratio,
            preserve=["decisions", "constraints", "errors"],
            drop=["reasoning_chains", "examples", "verbose"]
        )
    
    # Max iterations reached - raise instead of silent truncation
    raise CompressionFailedError(
        f"Could not compress to {max_tokens} tokens after {max_iterations} iterations"
    )


# Helper functions (each < 10 cognitive complexity per AP-2.1)
def _count_tokens(content: str, model: Llama) -> int:
    """Count tokens in content."""
    return len(model.tokenize(content.encode()))


def _calculate_compression_ratio(target: int, current: int) -> float:
    """Calculate required compression ratio with safety margin."""
    return (target / current) * 0.9  # 10% safety margin


async def _apply_compression(
    content: str,
    target_ratio: float,
    preserve: list[str],
    drop: list[str]
) -> str:
    """Apply compression using fast model."""
    return await compress_with_model(
        content=content,
        target_ratio=target_ratio,
        preserve=preserve,
        drop=drop
    )
```

### Context Window Mismatch Handling

In pipeline mode with different context windows:

```
Pipeline: llama-3.2-3b (8K) → qwen2.5-7b (32K) → phi-4 (16K)
```

| Handoff | Action |
|---------|--------|
| llama → qwen | Output fits, pass with trajectory |
| qwen → phi-4 | If output > 16K, compress before handoff |

**Rule:** Compress BEFORE calling next model, not after failure.

### Error Contamination Detection

From Guidelines: "Errors are sticky - once incorrect assumptions enter context, they bias subsequent reasoning."

```python
class ErrorContaminationDetector:
    """Detect and quarantine error-contaminated context between pipeline steps."""
    
    ERROR_MARKERS = [
        "I apologize",
        "I made an error",
        "That's incorrect",
        "hallucination",
        "I don't have information about",
    ]
    
    async def validate_handoff(self, state: HandoffState, output: str) -> ValidationResult:
        """Check for error contamination before passing to next model."""
        issues = []
        
        # Check for error markers in output
        for marker in self.ERROR_MARKERS:
            if marker.lower() in output.lower():
                issues.append(ErrorIssue(
                    type="error_marker_detected",
                    marker=marker,
                    severity="warning"
                ))
        
        # Check for contradiction with previous decisions
        for decision in state.decisions_made:
            if self._contradicts(output, decision):
                issues.append(ErrorIssue(
                    type="contradiction_detected",
                    decision=decision,
                    severity="high"
                ))
        
        # Check if error count is growing
        if len(state.active_errors) > state.current_step:
            issues.append(ErrorIssue(
                type="error_accumulation",
                count=len(state.active_errors),
                severity="high"
            ))
        
        return ValidationResult(
            valid=len([i for i in issues if i.severity == "high"]) == 0,
            issues=issues,
            recommendation="quarantine" if not valid else "proceed"
        )
```

**Actions on Detection:**

| Severity | Action |
|----------|--------|
| Warning | Log, continue with `active_errors` updated |
| High | Quarantine output, retry with fresh context |
| Critical | Abort pipeline, return partial result with error flag |

---

## Error Handling

### Exception Hierarchy

Custom exceptions enable proper retry logic and error classification:

```python
class InferenceServiceError(Exception):
    """Base exception for inference-service."""
    pass


class RetriableError(InferenceServiceError):
    """Transient errors that may succeed on retry."""
    
    def __init__(self, message: str, retry_after_ms: int = 1000):
        super().__init__(message)
        self.retry_after_ms = retry_after_ms


class NonRetriableError(InferenceServiceError):
    """Permanent errors that should not be retried."""
    pass


# Retriable Exceptions
class ModelBusyError(RetriableError):
    """Model is processing another request."""
    pass


class ModelLoadingError(RetriableError):
    """Model is still loading into memory."""
    pass


class TemporaryResourceError(RetriableError):
    """Temporary resource exhaustion (memory pressure)."""
    pass


# Non-Retriable Exceptions
class ContextBudgetExceededError(NonRetriableError):
    """Content cannot fit in context even after compression."""
    
    def __init__(self, current_tokens: int, budget: int):
        super().__init__(f"Context budget exceeded: {current_tokens}/{budget} tokens")
        self.current_tokens = current_tokens
        self.budget = budget


class CompressionFailedError(NonRetriableError):
    """Compression could not achieve target ratio after max iterations."""
    pass


class HandoffStateInvalidError(NonRetriableError):
    """HandoffState missing required fields for pipeline step."""
    pass


class ModelNotFoundError(NonRetriableError):
    """Requested model not available in configuration."""
    pass


class OrchestrationFailedError(NonRetriableError):
    """Multi-model orchestration failed (e.g., no consensus in ensemble)."""
    pass
```

### Error Response Format

Aligned with llm-gateway error schema:

```json
{
  "error": {
    "code": "CONTEXT_BUDGET_EXCEEDED",
    "message": "Content cannot fit in phi-4 context window (18000/16384 tokens)",
    "type": "non_retriable",
    "provider": "inference-service",
    "details": {
      "current_tokens": 18000,
      "budget": 16384,
      "model": "phi-4",
      "compression_attempted": true
    }
  }
}
```

### Saga Compensation Pattern

For pipeline failures, implement rollback to preserve partial progress:

```python
class PipelineSaga:
    """Saga orchestration for multi-model pipelines with compensation."""
    
    def __init__(self):
        self.steps: list[SagaStep] = []
        self.completed_steps: list[CompletedStep] = []
    
    async def execute(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        """Execute pipeline with compensation on failure."""
        state = HandoffState.from_request(request)
        
        for step in self.steps:
            try:
                result = await step.invoke(state)
                state = step.handle_reply(state, result)
                self.completed_steps.append(CompletedStep(
                    step=step,
                    state_snapshot=state.copy(),
                    result=result
                ))
            except RetriableError as e:
                # Retry this step
                await asyncio.sleep(e.retry_after_ms / 1000)
                continue
            except NonRetriableError as e:
                # Compensate and return partial result
                return await self._compensate(state, e)
        
        return self._build_response(state)
    
    async def _compensate(self, state: HandoffState, error: Exception) -> ChatCompletionResponse:
        """Roll back and return best partial result."""
        # Find last successful step with usable output
        for completed in reversed(self.completed_steps):
            if completed.result.is_usable:
                return ChatCompletionResponse(
                    choices=[Choice(
                        message=Message(
                            role="assistant",
                            content=completed.result.output
                        ),
                        finish_reason="partial"
                    )],
                    orchestration=OrchestrationMetadata(
                        mode="pipeline",
                        completed_steps=len(self.completed_steps),
                        total_steps=len(self.steps),
                        error=str(error),
                        partial=True
                    )
                )
        
        # No usable partial result
        raise OrchestrationFailedError(f"Pipeline failed at step 1: {error}")
```

---

## Caching Strategy

### Cache Separation

| Cache Type | Purpose | Scope | Storage |
|------------|---------|-------|---------|
| **Operational Cache** | Handoffs, compression, budgets | Per-request, ephemeral | In-memory / Redis |
| **Conversation Store** | Full history, reporting | Persistent, append-only | audit-service (separate) |

### What inference-service Caches

#### 1. Prompt Cache (Tokenized System Prompts)

```python
class PromptCache:
    """Cache tokenized prompts since they rarely change."""
    
    def __init__(self, model: Llama):
        self.model = model
        self._cache = {}
    
    def get_tokenized(self, text: str) -> list[int]:
        key = hashlib.md5(text.encode()).hexdigest()
        if key not in self._cache:
            self._cache[key] = self.model.tokenize(text.encode())
        return self._cache[key]
```

#### 2. Handoff State Cache

```python
import asyncio
from typing import Optional

class HandoffCache:
    """Cache structured handoff state between pipeline steps.
    
    Note: Uses per-resource locks per CODING_PATTERNS_ANALYSIS AP-10.1
    to prevent race conditions in async context.
    """
    
    def __init__(self, redis_client: Optional[redis.Redis] = None):
        self.redis = redis_client
        self._local: dict[str, str] = {}  # Fallback for dev
        self._locks: dict[str, asyncio.Lock] = {}  # Per-resource locks (AP-10.1)
    
    def _get_lock(self, key: str) -> asyncio.Lock:
        """Get or create lock for specific key."""
        if key not in self._locks:
            self._locks[key] = asyncio.Lock()
        return self._locks[key]
    
    async def store(self, state: HandoffState, ttl: int = 3600) -> None:
        key = f"handoff:{state.request_id}"
        async with self._get_lock(key):
            data = state.model_dump_json()
            if self.redis:
                await self.redis.setex(key, ttl, data)  # Use async Redis
            else:
                self._local[key] = data
    
    async def get(self, request_id: str) -> Optional[HandoffState]:
        key = f"handoff:{request_id}"
        async with self._get_lock(key):
            if self.redis:
                data = await self.redis.get(key)
            else:
                data = self._local.get(key)
            return HandoffState.model_validate_json(data) if data else None
```

#### 3. Compression Result Cache

```python
class CompressionCache:
    """Cache compressed versions of content."""
    
    def get_key(self, content: str, target_tokens: int) -> str:
        content_hash = hashlib.md5(content.encode()).hexdigest()
        return f"{content_hash}:{target_tokens}"
    
    def get(self, content: str, target_tokens: int) -> Optional[str]:
        return self._cache.get(self.get_key(content, target_tokens))
    
    def store(self, content: str, target_tokens: int, compressed: str):
        self._cache[self.get_key(content, target_tokens)] = compressed
```

#### 4. Semantic Cache (Optional, Future Enhancement)

From Guidelines Segment 43: "Semantic caching with vector similarity for query reuse."

```python
class SemanticCache:
    """Cache LLM responses for semantically similar queries.
    
    Note: User-specific queries should NOT be cached.
    Time-sensitive queries require fresh generation.
    """
    
    def __init__(
        self,
        embedding_model: str = "all-MiniLM-L6-v2",
        similarity_threshold: float = 0.95,  # High threshold to avoid false positives
        ttl_seconds: int = 3600,
    ):
        self.embedder = SentenceTransformer(embedding_model)
        self.similarity_threshold = similarity_threshold
        self.ttl_seconds = ttl_seconds
        self._cache: dict[str, CacheEntry] = {}  # In production: vector DB
    
    async def get(self, query: str, model_id: str) -> Optional[str]:
        """Check for semantically similar cached response."""
        query_embedding = self.embedder.encode(query)
        
        for key, entry in self._cache.items():
            if entry.model_id != model_id:
                continue
            if entry.is_expired():
                continue
            
            similarity = cosine_similarity(query_embedding, entry.query_embedding)
            if similarity >= self.similarity_threshold:
                return entry.response
        
        return None
    
    async def store(
        self,
        query: str,
        response: str,
        model_id: str,
        is_user_specific: bool = False,
        is_time_sensitive: bool = False,
    ) -> None:
        """Store response with metadata."""
        # Don't cache user-specific or time-sensitive queries
        if is_user_specific or is_time_sensitive:
            return
        
        self._cache[self._make_key(query, model_id)] = CacheEntry(
            query_embedding=self.embedder.encode(query),
            response=response,
            model_id=model_id,
            created_at=datetime.utcnow(),
            ttl_seconds=self.ttl_seconds,
        )
```

**When to Use:**

| Use Semantic Cache | Don't Use |
|--------------------|----------|
| General knowledge queries | User-specific data |
| Code explanations | Current time/date |
| Language translations | Personalized responses |
| Documentation lookups | Queries with user PII |

### Cache Invalidation Strategy

From Code Reference Engine patterns: track model version in cache keys.

```python
class CacheInvalidator:
    """Invalidate caches when model configuration changes."""
    
    def __init__(self, caches: list[InferenceCache]):
        self.caches = caches
        self._model_versions: dict[str, str] = {}
    
    def on_model_loaded(self, model_id: str, model_hash: str) -> None:
        """Track model version on load."""
        previous = self._model_versions.get(model_id)
        self._model_versions[model_id] = model_hash
        
        if previous and previous != model_hash:
            # Model changed - invalidate caches
            self._invalidate_for_model(model_id)
    
    def on_config_change(self, old_config: str, new_config: str) -> None:
        """Invalidate all caches on config change."""
        if old_config != new_config:
            for cache in self.caches:
                cache.clear()
    
    def _invalidate_for_model(self, model_id: str) -> None:
        """Invalidate cache entries for specific model."""
        for cache in self.caches:
            cache.invalidate_by_model(model_id)


class InferenceCache(ABC):
    """Abstract base for all inference caches."""
    
    @abstractmethod
    def get(self, key: str) -> Optional[Any]: ...
    
    @abstractmethod
    def store(self, key: str, value: Any, ttl: int = 3600) -> None: ...
    
    @abstractmethod
    def invalidate_by_model(self, model_id: str) -> None: ...
    
    @abstractmethod
    def clear(self) -> None: ...
    
    @abstractmethod
    def size(self) -> int: ...
```

**Invalidation Triggers:**

| Trigger | Action |
|---------|--------|
| Model file changed (hash mismatch) | Invalidate model-specific entries |
| Config preset changed | Clear all caches |
| Model unloaded | Invalidate model-specific entries |
| TTL expired | Entry auto-expires |
| Memory pressure | LRU eviction |

### What inference-service Does NOT Cache

| Don't Cache | Why |
|-------------|-----|
| Full model outputs | Too big, context-specific |
| Conversation history | Changes every turn, owned by audit-service |
| Raw user queries | Rarely repeat exactly |
| Error details | Must be fresh |
| User-specific content | Privacy, personalization |
| Time-sensitive queries | Stale data risk |

### KV Cache (Model-Level)

llama-cpp-python manages KV cache internally:

| Scenario | KV Cache Benefit |
|----------|------------------|
| Same model, continuing conversation | ✅ Reuse cache |
| Same model, new conversation | ❌ Cold start |
| Different model in pipeline | ❌ Incompatible |
| Model swap (sequential) | ❌ Cache lost on unload |

**Mac implication:** Model swapping loses KV cache. Cache structured handoff state instead.

### Response Metadata for Upstream Capture

inference-service returns metadata that llm-gateway/audit-service captures:

```json
{
  "id": "chatcmpl-abc123",
  "choices": [...],
  "usage": {
    "prompt_tokens": 150,
    "completion_tokens": 500,
    "total_tokens": 650
  },
  "inference_metadata": {
    "config": "D3",
    "orchestration_mode": "debate",
    "models_used": ["phi-4", "deepseek-r1-7b"],
    "handoff_steps": 3,
    "compression_applied": true,
    "context_utilization": 0.72
  }
}
```

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
INFERENCE_MODEL_ROLES='{"llama-3.2-3b":["fast"],"qwen2.5-7b":["coder","primary"],"deepseek-r1-7b":["thinker"],"granite-8b-code-128k":["coder","longctx"],"phi-4":["primary","thinker","coder"],"phi-3-medium-128k":["longctx","thinker"],"gpt-oss-20b":["primary","thinker"],"granite-20b-code":["coder","thinker"]}'

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

# === Mac-Compatible Models (~32GB total) ===

# llama-3.2-3b (2.0GB) - Fast drafts
huggingface-cli download hugging-quants/Llama-3.2-3B-Instruct-Q4_K_M-GGUF llama-3.2-3b-instruct-q4_k_m.gguf --local-dir llama-3.2-3b

# qwen2.5-7b (4.4GB) - Code generation
huggingface-cli download Qwen/Qwen2.5-7B-Instruct-GGUF qwen2.5-7b-instruct-q4_k_m.gguf --local-dir qwen2.5-7b

# deepseek-r1-7b (4.4GB) - Chain-of-thought reasoning
huggingface-cli download unsloth/DeepSeek-R1-Distill-Qwen-7B-GGUF DeepSeek-R1-Distill-Qwen-7B-Q4_K_M.gguf --local-dir deepseek-r1-7b

# granite-8b-code-128k (4.5GB) - Full-file code analysis, 128K context
huggingface-cli download mradermacher/granite-8b-code-instruct-128k-GGUF granite-8b-code-instruct-128k.Q4_K_M.gguf --local-dir granite-8b-code-128k

# phi-4 (7.9GB) - General reasoning
huggingface-cli download microsoft/phi-4-gguf phi-4-Q4_K_S.gguf --local-dir phi-4

# phi-3-medium-128k (8.6GB) - Long document processing
huggingface-cli download bartowski/Phi-3-medium-128k-instruct-GGUF Phi-3-medium-128k-instruct-Q4_K_M.gguf --local-dir phi-3-medium-128k

# === Server-Only Models (~24GB additional) ===

# gpt-oss-20b (11.6GB) - High-capacity reasoning
huggingface-cli download unsloth/gpt-oss-20b-GGUF gpt-oss-20b-Q4_K_M.gguf --local-dir gpt-oss-20b

# granite-20b-code (12.8GB) - Complex code tasks
huggingface-cli download mradermacher/granite-20b-code-instruct-GGUF granite-20b-code-instruct.Q4_K_M.gguf --local-dir granite-20b-code
```

### Total Storage

| Category | Models | Size |
|----------|--------|------|
| Mac-compatible | 6 models | ~32GB |
| Server-only | 2 models | ~24GB |
| **Total** | **8 models** | **~56GB** |

---

## 15. Observability

### OpenTelemetry Tracing (OBS-11)

The inference-service now supports distributed tracing via OpenTelemetry, enabling request correlation across the Kitchen Brigade services.

#### Configuration

| Environment Variable | Default | Description |
|---------------------|---------|-------------|
| `OTEL_EXPORTER_OTLP_ENDPOINT` | `http://localhost:4317` | OTLP collector endpoint |
| `OTEL_SERVICE_NAME` | `inference-service` | Service name in traces |
| `OTEL_ENABLED` | `true` | Enable/disable tracing |

#### Trace Propagation

```
External Request
      │
      ▼
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   ai-agents     │────▶│   CMS:8086      │────▶│ inference:8085  │
│   trace_id: X   │     │   trace_id: X   │     │   trace_id: X   │
│   span: agent   │     │   span: proxy   │     │   span: infer   │
└─────────────────┘     └─────────────────┘     └─────────────────┘
                                                        │
                                                        ▼
                                               ┌─────────────────┐
                                               │    Jaeger UI    │
                                               │ Trace Viewer    │
                                               └─────────────────┘
```

#### Files

| File | Purpose |
|------|---------|
| `src/observability/__init__.py` | Module exports |
| `src/observability/tracing.py` | OpenTelemetry setup, span creation |

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | 2025-12-27 | Initial architecture design |
| 1.1.0 | 2025-12-27 | Added gpt-oss-20b, granite-8b-code-128k, granite-20b-code; expanded configs to 41 total |
| 1.2.0 | 2025-12-27 | Added Context Management and Caching Strategy sections |
| 1.3.0 | 2025-12-27 | Document Analysis Phase validation: Added Error Handling section (exception hierarchy, saga compensation), Error Contamination Detection, Semantic Cache Layer, Cache Invalidation Strategy, Cache Interface Abstraction |
| 1.3.1 | 2025-12-27 | Anti-pattern fixes from CODING_PATTERNS_ANALYSIS: AP-1.5 (HandoffState default_factory), AP-2.1 (fit_to_budget iterative), AP-10.1 (HandoffCache asyncio.Lock) |
| 1.4.0 | 2025-12-31 | Implementation phase updates |
| 1.5.0 | 2026-01-27 | Added Observability section (OBS-11), CMS proxy routing diagram, updated Kitchen Brigade positioning |
