# Model Library and Configuration Guide

> **Version:** 1.0.0  
> **Created:** 2025-12-28  
> **Reference:** [ARCHITECTURE.md](ARCHITECTURE.md), [WBS.md](WBS.md)

## Overview

This document describes all available models, their characteristics, environment variables for loading them, and the configuration presets that combine them for different orchestration modes.

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Available Models](#available-models)
3. [Environment Variables](#environment-variables)
4. [Configuration Presets](#configuration-presets)
5. [Orchestration Modes](#orchestration-modes)
6. [Role Definitions](#role-definitions)
7. [Hardware Recommendations](#hardware-recommendations)

---

## Quick Start

```bash
# Single model (simplest)
export INFERENCE_CONFIG=S1
export INFERENCE_MODELS_DIR="/Volumes/NO NAME/LLMs/models"
python -m uvicorn src.main:app --host 0.0.0.0 --port 8085

# Dual model critique (recommended for Mac 16GB)
export INFERENCE_CONFIG=D4
export INFERENCE_ORCHESTRATION_MODE=critique
```

---

## Available Models

### Model Summary Table

| Model ID | Name | Size | Context | Primary Role | Hardware |
|----------|------|------|---------|--------------|----------|
| `phi-4` | Microsoft Phi-4 | 8.4 GB | 16K | General reasoning | Mac 16GB ✓ |
| `deepseek-r1-7b` | DeepSeek R1 Distill 7B | 4.7 GB | 32K | Chain-of-thought | Mac 16GB ✓ |
| `qwen2.5-7b` | Qwen 2.5 7B Instruct | 4.5 GB | 32K | Code generation | Mac 16GB ✓ |
| `llama-3.2-3b` | Llama 3.2 3B Instruct | 2.0 GB | 8K | Fast responses | Mac 16GB ✓ |
| `phi-3-medium-128k` | Phi-3 Medium 128K | 8.6 GB | 128K | Long documents | Mac 16GB ✓ |
| `granite-8b-code-128k` | IBM Granite 8B Code 128K | 4.5 GB | 128K | Full-file code | Mac 16GB ✓ |
| `gpt-oss-20b` | GPT-OSS 20B | 11.6 GB | 32K | High-capacity | Server only |
| `granite-20b-code` | IBM Granite 20B Code | 12.8 GB | 8K | High-capacity code | Server only |

### Detailed Model Specifications

#### phi-4 (Microsoft Phi-4)
```yaml
File: phi-4-Q4_K_S.gguf
Size: 8.4 GB
Context Length: 16,384 tokens
Quantization: Q4_K_S
Roles: [primary, thinker, coder]
GPU Layers: -1 (all)
```
**Best for:** General reasoning, summarization, coding tasks, validation in pipelines.

---

#### deepseek-r1-7b (DeepSeek R1 Distill 7B)
```yaml
File: DeepSeek-R1-Distill-Qwen-7B-Q4_K_M.gguf
Size: 4.7 GB
Context Length: 32,768 tokens
Quantization: Q4_K_M
Roles: [thinker]
GPU Layers: -1 (all)
```
**Best for:** Chain-of-thought reasoning, complex analysis, critique phase in orchestration.

---

#### qwen2.5-7b (Qwen 2.5 7B Instruct)
```yaml
File: qwen2.5-7b-instruct-q4_k_m.gguf
Size: 4.5 GB
Context Length: 32,768 tokens
Quantization: Q4_K_M
Roles: [coder, primary, fast]
GPU Layers: -1 (all)
```
**Best for:** Code generation, technical tasks, refining code in pipelines.

---

#### llama-3.2-3b (Llama 3.2 3B Instruct)
```yaml
File: llama-3.2-3b-instruct-q4_k_m.gguf
Size: 2.0 GB
Context Length: 8,192 tokens
Quantization: Q4_K_M
Roles: [fast]
GPU Layers: -1 (all)
```
**Best for:** Fast drafts, simple queries, classification, as drafter in pipelines.

---

#### phi-3-medium-128k (Phi-3 Medium 128K Instruct)
```yaml
File: Phi-3-medium-128k-instruct-Q4_K_M.gguf
Size: 8.6 GB
Context Length: 131,072 tokens (128K)
Quantization: Q4_K_M
Roles: [longctx, thinker]
GPU Layers: -1 (all)
```
**Best for:** Long document analysis, full-file processing, RAG applications.

---

#### granite-8b-code-128k (IBM Granite 8B Code 128K)
```yaml
File: granite-8b-code-instruct-128k.Q4_K_M.gguf
Size: 4.5 GB
Context Length: 131,072 tokens (128K)
Quantization: Q4_K_M
Roles: [coder, longctx]
GPU Layers: -1 (all)
```
**Best for:** Full-file code analysis, large codebase review, code-focused long context.

---

#### gpt-oss-20b (GPT-OSS 20B) ⚠️ Server Only
```yaml
File: gpt-oss-20b-Q4_K_M.gguf
Size: 11.6 GB
Context Length: 32,768 tokens
Quantization: Q4_K_M
Roles: [primary, thinker]
GPU Layers: -1 (all)
Requirements: >16GB RAM/VRAM
```
**Best for:** High-capacity general reasoning (server deployment only).

---

#### granite-20b-code (IBM Granite 20B Code) ⚠️ Server Only
```yaml
File: granite-20b-code-instruct.Q4_K_M.gguf
Size: 12.8 GB
Context Length: 8,192 tokens
Quantization: Q4_K_M
Roles: [coder, thinker]
GPU Layers: -1 (all)
Requirements: >16GB RAM/VRAM
```
**Best for:** High-capacity code generation (server deployment only).

---

## Environment Variables

All inference-service settings use the `INFERENCE_` prefix.

### Core Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `INFERENCE_PORT` | `8085` | HTTP server port (1-65535) |
| `INFERENCE_HOST` | `0.0.0.0` | HTTP server bind address |
| `INFERENCE_ENVIRONMENT` | `development` | `development`, `staging`, `production` |
| `INFERENCE_LOG_LEVEL` | `INFO` | `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL` |

### Model Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `INFERENCE_MODELS_DIR` | `/models` | Path to GGUF model files directory |
| `INFERENCE_GPU_LAYERS` | `-1` | Layers on GPU (`-1` = all, Metal on Mac) |
| `INFERENCE_BACKEND` | `llamacpp` | Inference backend (`llamacpp` or `vllm`) |
| `INFERENCE_CONFIG` | `S1` | Configuration preset ID (see below) |

### Orchestration Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `INFERENCE_ORCHESTRATION_MODE` | `single` | `single`, `critique`, `debate`, `ensemble`, `pipeline` |
| `INFERENCE_MAX_CONCURRENT_REQUESTS` | `5` | Maximum concurrent inference requests |
| `INFERENCE_REQUEST_TIMEOUT` | `30` | Request timeout in seconds |
| `INFERENCE_CACHE_ENABLED` | `false` | Enable prompt/response caching |

### Example Configuration

```bash
# Mac 16GB - Dual model critique mode
export INFERENCE_PORT=8085
export INFERENCE_HOST=0.0.0.0
export INFERENCE_MODELS_DIR="/Volumes/NO NAME/LLMs/models"
export INFERENCE_GPU_LAYERS=-1
export INFERENCE_CONFIG=D4
export INFERENCE_ORCHESTRATION_MODE=critique
export INFERENCE_LOG_LEVEL=INFO
export INFERENCE_MAX_CONCURRENT_REQUESTS=2
```

---

## Configuration Presets

Configuration presets define which models to load together and how they should orchestrate.

### Single Model Configs (S)

| Config | Name | Models | Size | Mode | Hardware |
|--------|------|--------|------|------|----------|
| **S1** | Phi-4 Solo | phi-4 | 8.4 GB | single | Mac 16GB (safe) |
| **S2** | DeepSeek Solo | deepseek-r1-7b | 4.7 GB | single | Mac 16GB (light) |
| **S3** | Qwen Solo | qwen2.5-7b | 4.5 GB | single | Mac 16GB (light) |
| **S4** | Llama Fast Solo | llama-3.2-3b | 2.0 GB | single | Mac 16GB (minimal) |
| **S5** | Phi-3 Long Context | phi-3-medium-128k | 8.6 GB | single | Mac 16GB (safe) |
| **S6** | Granite Code 128K | granite-8b-code-128k | 4.5 GB | single | Mac 16GB (comfortable) |
| **S7** | GPT-OSS 20B | gpt-oss-20b | 11.6 GB | single | Server (24GB+) |
| **S8** | Granite 20B Code | granite-20b-code | 12.8 GB | single | Server (24GB+) |

---

### Dual Model Configs (D)

| Config | Name | Models | Size | Mode | Description |
|--------|------|--------|------|------|-------------|
| **D1** | Quality + Speed | phi-4, llama-3.2-3b | 10.4 GB | pipeline | Fast drafts → quality validation |
| **D2** | Reasoning + Code | phi-4, qwen2.5-7b | 12.9 GB | critique | General reasoning with code critique |
| **D3** | Reasoning Debate | phi-4, deepseek-r1-7b | 13.1 GB | debate | Two reasoners debate |
| **D4** | Thinking + Code | deepseek-r1-7b, qwen2.5-7b | 9.2 GB | critique | Code with chain-of-thought critique |
| **D5** | Code + Fast | qwen2.5-7b, llama-3.2-3b | 6.5 GB | pipeline | Fast drafts → code refinement |
| **D6** | Long + Fast | phi-3-medium-128k, llama-3.2-3b | 9.5 GB | pipeline | Quick drafts → long expansion |
| **D7** | Thinking + Fast | deepseek-r1-7b, llama-3.2-3b | 6.7 GB | critique | Fast gen → thinking critique |
| **D8** | General + Long | phi-4, phi-3-medium-128k | 15.9 GB | critique | Long doc gen → general critique |
| **D9** | Long + Code | phi-3-medium-128k, qwen2.5-7b | 12.0 GB | critique | Long doc → code review |
| **D10** | Long Context Debate | phi-3-medium-128k, deepseek-r1-7b | 12.2 GB | debate | Long document debate |

#### D4 Role Assignment (Recommended for Mac 16GB)
```yaml
D4:
  name: "Thinking + Code"
  models: [deepseek-r1-7b, qwen2.5-7b]
  total_size_gb: 9.2
  orchestration_mode: critique
  roles:
    qwen2.5-7b: generator    # Generates initial code
    deepseek-r1-7b: critic   # Chain-of-thought critique
```

---

### Triple Model Configs (T)

| Config | Name | Models | Size | Mode |
|--------|------|--------|------|------|
| **T1** | Pipeline: Draft→Code→Validate | phi-4, qwen2.5-7b, llama-3.2-3b | 14.9 GB | pipeline |
| **T2** | Reasoning Ensemble | phi-4, deepseek-r1-7b, llama-3.2-3b | 15.1 GB | ensemble |
| **T3** | Pipeline: Think→Code→Draft | deepseek-r1-7b, qwen2.5-7b, llama-3.2-3b | 11.2 GB | pipeline |
| **T4** | Full Reasoning Debate | phi-4, deepseek-r1-7b, qwen2.5-7b | 17.6 GB | debate |
| **T5** | Long Context Pipeline | phi-3-medium-128k, qwen2.5-7b, llama-3.2-3b | 14.0 GB | pipeline |
| **T6** | General+Long+Fast | phi-4, phi-3-medium-128k, llama-3.2-3b | 17.9 GB | pipeline |
| **T7** | General+Long+Code Ensemble | phi-4, phi-3-medium-128k, qwen2.5-7b | 20.4 GB | ensemble |
| **T8** | Reasoning Trio Debate | phi-4, phi-3-medium-128k, deepseek-r1-7b | 20.6 GB | debate |
| **T9** | Long+Think+Code Ensemble | phi-3-medium-128k, deepseek-r1-7b, qwen2.5-7b | 16.7 GB | ensemble |
| **T10** | Long Context Think Pipeline | phi-3-medium-128k, deepseek-r1-7b, llama-3.2-3b | 14.2 GB | pipeline |

#### T1 Role Assignment (Full Pipeline)
```yaml
T1:
  name: "Pipeline: Draft → Code → Validate"
  models: [phi-4, qwen2.5-7b, llama-3.2-3b]
  total_size_gb: 14.9
  orchestration_mode: pipeline
  roles:
    llama-3.2-3b: drafter    # Fast initial draft
    qwen2.5-7b: refiner      # Code-aware refinement
    phi-4: validator         # Final quality check
```

---

### Quad Model Configs (Q) - Server Only

| Config | Name | Models | Size | Mode |
|--------|------|--------|------|------|
| **Q1** | Full Capability | phi-4, deepseek-r1-7b, qwen2.5-7b, llama-3.2-3b | 19.6 GB | ensemble |
| **Q2** | Full Pipeline+Long | phi-4, phi-3-medium-128k, qwen2.5-7b, llama-3.2-3b | 22.4 GB | pipeline |
| **Q3** | General+Long+Think+Fast | phi-4, phi-3-medium-128k, deepseek-r1-7b, llama-3.2-3b | 22.6 GB | ensemble |
| **Q4** | Full Reasoning Debate | phi-4, phi-3-medium-128k, deepseek-r1-7b, qwen2.5-7b | 25.1 GB | debate |
| **Q5** | Specialist Ensemble | phi-3-medium-128k, deepseek-r1-7b, qwen2.5-7b, llama-3.2-3b | 18.7 GB | ensemble |

---

### Quint Model Configs (P) - Server Only (48GB+ VRAM)

| Config | Name | Models | Size | Mode |
|--------|------|--------|------|------|
| **P1** | Maximum Ensemble | All 5 models | 27.1 GB | ensemble |
| **P2** | Full Pipeline | All 5 models | 27.1 GB | pipeline |
| **P3** | Maximum Debate | All 5 models | 27.1 GB | debate |

#### P2 Role Assignment (Full Five-Stage Pipeline)
```yaml
P2:
  name: "Full Pipeline"
  models: [phi-4, phi-3-medium-128k, deepseek-r1-7b, qwen2.5-7b, llama-3.2-3b]
  total_size_gb: 27.1
  orchestration_mode: pipeline
  roles:
    llama-3.2-3b: drafter      # Stage 1: Quick draft
    qwen2.5-7b: coder          # Stage 2: Code enhancement
    deepseek-r1-7b: thinker    # Stage 3: Reasoning pass
    phi-4: validator           # Stage 4: Quality validation
    phi-3-medium-128k: expander # Stage 5: Long context expansion
```

---

## Orchestration Modes

### Single Mode
```
Request → Model → Response
```
- Direct pass-through to one model
- Lowest latency
- Best for: Simple queries, fast responses

### Critique Mode
```
Request → Generator → Critic → [Revise] → Response
```
- Generator creates initial response
- Critic analyzes and provides feedback
- Optional revision cycle
- Best for: Code review, quality improvement

### Pipeline Mode
```
Request → Drafter → Refiner → Validator → [Expander] → Response
```
- Sequential processing through stages
- Each model specializes in one phase
- Saga compensation on failure (returns partial result)
- Best for: Complex multi-step tasks

### Debate Mode
```
Request → [Model A, Model B] (parallel) → Reconciler → Response
```
- Multiple models generate in parallel
- Reconciler synthesizes final answer
- Agreement percentage in metadata
- Best for: High-stakes decisions, consensus building

### Ensemble Mode
```
Request → [All Models] (parallel) → Synthesizer → Response
```
- All models vote in parallel
- Consensus score calculated
- Disagreements flagged in metadata
- Best for: Maximum accuracy, diverse perspectives

---

## Role Definitions

| Role | Description | Task Types | Best Models |
|------|-------------|------------|-------------|
| **primary** | Default/general purpose | general, summarize, explain, chat | phi-4, qwen2.5-7b |
| **fast** | Quick responses, simple tasks | simple, quick, classify, extract | llama-3.2-3b, qwen2.5-7b |
| **coder** | Code generation and review | code, debug, review, refactor, test | qwen2.5-7b, granite-8b-code-128k |
| **thinker** | Complex reasoning and analysis | analyze, reason, compare, debate, plan | deepseek-r1-7b, phi-4 |
| **longctx** | Long document processing | document, summarize_long, rag | phi-3-medium-128k, granite-8b-code-128k |

### Pipeline Role Assignments

| Pipeline Role | Description | Best Models |
|---------------|-------------|-------------|
| **drafter** | Fast initial generation | llama-3.2-3b |
| **refiner** | Improve/enhance draft | qwen2.5-7b |
| **validator** | Quality check/verification | phi-4, deepseek-r1-7b |
| **expander** | Add detail/long context | phi-3-medium-128k |
| **coder** | Code-specific enhancement | qwen2.5-7b, granite-8b-code-128k |
| **thinker** | Reasoning pass | deepseek-r1-7b |

---

## Hardware Recommendations

### Mac 16GB Configurations

| Category | Configs | Max Size | Notes |
|----------|---------|----------|-------|
| **Light** (8GB headroom) | S2, S3, S4, D5, D7 | 8 GB | Other apps running |
| **Medium** (5GB headroom) | S1, S5, S6, D1, D4, D9, D10 | 13 GB | VS Code + service |
| **Full** (2GB headroom) | D3, D8, T1, T3, T5, T10 | 15 GB | Dedicated to inference |

### Server Configurations

| Category | Configs | VRAM Required |
|----------|---------|---------------|
| **24GB VRAM** | S7, S8, T4, T6 | 24 GB |
| **32GB VRAM** | T7, T8, T9, Q1-Q5 | 32 GB |
| **48GB+ VRAM** | P1, P2, P3 | 48 GB+ |

### Recommended Starting Configurations

| Use Case | Config | Why |
|----------|--------|-----|
| **Quick Testing** | S4 | Minimal footprint, fast responses |
| **General Dev** | S1 | Best single-model quality |
| **Code + Quality** | D4 | Code gen + thinking critique |
| **Maximum Local Quality** | T1 | Full pipeline on Mac 16GB |
| **Production Server** | P1 | Five-way consensus, highest accuracy |

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | 2025-12-28 | Initial model library documentation |
