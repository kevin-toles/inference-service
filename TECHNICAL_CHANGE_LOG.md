# inference-service: Technical Change Log

**Purpose**: Documents architectural decisions, API changes, and significant updates to the Inference Service (local LLM inference with Metal/CUDA GPU acceleration).

---

## Changelog

### 2026-01-18: WBS-GPU Multi-GPU Documentation (CL-005)

**Summary**: Added comprehensive GPU allocation documentation and multi-GPU support configuration.

**GPU Modes:**

| Mode | Config | Use Case |
|------|--------|----------|
| Metal (macOS) | `INFERENCE_GPU_LAYERS=-1` | Apple Silicon acceleration |
| CUDA | `CUDA_VISIBLE_DEVICES=0,1` | NVIDIA multi-GPU |
| CPU | `INFERENCE_GPU_LAYERS=0` | Fallback/testing |

**Files Changed:**

| File | Changes |
|------|---------|
| `docs/GPU_ALLOCATION.md` | New: GPU configuration guide |
| `config/presets.yaml` | Multi-GPU preset options |

**Cross-References:**
- WBS-GPU: GPU acceleration work package

---

### 2026-01-15: WBS-LOG0 Structured Logging (CL-004)

**Summary**: Added structured JSON logging with correlation ID support.

**Files Changed:**

| File | Changes |
|------|---------|
| `src/core/logging.py` | Structured logging |
| `src/api/middleware.py` | Correlation ID handling |

---

### 2026-01-12: Native Startup Script Update (CL-003)

**Summary**: Updated run_native.sh with proper environment variable handling for Metal GPU.

**Environment Variables:**

| Variable | Default | Purpose |
|----------|---------|---------|
| `INFERENCE_GPU_LAYERS` | `-1` | All layers on GPU |
| `INFERENCE_MODELS_DIR` | `/ai-models/models` | Model storage path |
| `INFERENCE_CONFIG_DIR` | `./config` | Config directory |

**Files Changed:**

| File | Changes |
|------|---------|
| `run_native.sh` | Environment handling |

---

### 2026-01-10: Model Configuration Update (CL-002)

**Summary**: Updated models.yaml with new Qwen3 and DeepSeek-R1 model definitions.

**Models Added/Updated:**

| Model | Size | Architecture |
|-------|------|--------------|
| `qwen3-8b` | 4.7GB | Dense |
| `qwen3-coder-30b-a3b` | 14GB | MoE |
| `deepseek-r1-7b` | 4.7GB | Dense |
| `deepseek-r1-14b` | 8.5GB | Dense |

**Files Changed:**

| File | Changes |
|------|---------|
| `config/models.yaml` | New model definitions |
| `config/presets.yaml` | New preset combinations |

---

### 2026-01-01: Initial Inference Service (CL-001)

**Summary**: Initial service with llama.cpp backend for local LLM inference.

**Core Components:**

| Component | Purpose |
|-----------|---------|
| `LlamaCppBackend` | llama.cpp Python bindings |
| `ModelManager` | Model loading/unloading |
| `PresetManager` | Preset configurations |
| `ContextManager` | Context window handling |

**API Endpoints:**

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/health` | GET | Health check |
| `/v1/chat/completions` | POST | OpenAI-compatible chat |
| `/v1/models` | GET | List available models |
| `/v1/models/{model}/load` | POST | Load specific model |
| `/v1/models/{model}/unload` | POST | Unload model |

**Configuration:**

| Setting | Default | Purpose |
|---------|---------|---------|
| `INFERENCE_PORT` | 8085 | Service port |
| `INFERENCE_GPU_LAYERS` | -1 | GPU layer count |
| `INFERENCE_BACKEND` | `llamacpp` | Backend type |

**Cross-References:**
- INFERENCE_SERVICE_WBS.md: Full implementation WBS
