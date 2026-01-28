# Inference Service - Comprehensive Inventory

**Service**: inference-service  
**Port**: 8085  
**Version**: 0.1.0  
**Last Updated**: January 27, 2026

---

## 1. Module Inventory

### 1.1 Directory Structure

```
src/
├── main.py                          # FastAPI application entrypoint
├── __init__.py
├── py.typed
├── api/
│   ├── __init__.py
│   ├── error_handlers.py            # Exception handlers & error response schema
│   └── routes/
│       ├── __init__.py
│       ├── completions.py           # OpenAI-compatible chat completions
│       ├── health.py                # Liveness & readiness probes
│       ├── models.py                # Model lifecycle management
│       └── vision.py                # VLM image classification endpoints
├── core/
│   ├── __init__.py
│   ├── config.py                    # Pydantic settings (INFERENCE_* env vars)
│   ├── constants.py                 # Container paths & defaults
│   ├── exceptions.py                # Exception hierarchy (554 lines)
│   └── logging.py                   # Structured logging with structlog
├── models/
│   ├── __init__.py
│   ├── requests.py                  # ChatCompletionRequest & Message schemas
│   └── responses.py                 # ChatCompletionResponse & streaming chunks
├── observability/
│   ├── __init__.py
│   └── tracing.py                   # OpenTelemetry distributed tracing
├── orchestration/
│   ├── __init__.py
│   ├── context.py                   # HandoffState & context budget management
│   ├── orchestrator.py              # Multi-model orchestration dispatcher
│   ├── saga.py                      # Saga compensation pattern for pipelines
│   └── modes/
│       ├── __init__.py
│       ├── single.py                # Single model pass-through (implemented)
│       ├── critique.py              # Generate-then-critique (stub)
│       ├── debate.py                # Parallel-then-reconcile (stub)
│       ├── ensemble.py              # Vote-and-synthesize (stub)
│       └── pipeline.py              # Sequential stages (stub)
├── providers/
│   ├── __init__.py
│   ├── base.py                      # InferenceProvider ABC & ModelMetadata
│   ├── llamacpp.py                  # LlamaCpp provider (GGUF/Metal)
│   ├── moondream.py                 # Moondream 2 VLM provider
│   ├── deepseek_vl.py               # DeepSeek VL2 provider (legacy)
│   └── backends/
│       ├── __init__.py
│       ├── device_backend.py        # Device dtype compatibility ABC
│       ├── factory.py               # DeviceBackendFactory (MPS/CUDA/CPU)
│       └── siglip_attention_patch.py # SigLIP attention MPS fix
└── services/
    ├── __init__.py
    ├── cache.py                     # InferenceCache ABC, PromptCache, HandoffCache
    ├── model_manager.py             # Model lifecycle & preset loading
    └── queue_manager.py             # Request queuing & concurrency control
```

### 1.2 Module Purposes

#### Routers (FastAPI Routers)

| Module | Purpose | Lines |
|--------|---------|-------|
| [routes/completions.py](src/api/routes/completions.py) | OpenAI-compatible `/v1/chat/completions` endpoint | 170 |
| [routes/health.py](src/api/routes/health.py) | K8s liveness (`/health`) and readiness (`/health/ready`) probes | 207 |
| [routes/models.py](src/api/routes/models.py) | Model lifecycle: list, load, unload (`/v1/models/*`) | 243 |
| [routes/vision.py](src/api/routes/vision.py) | VLM image classification (`/api/v1/vision/*`) | 431 |

#### Model Loaders / Providers

| Module | Purpose | Lines |
|--------|---------|-------|
| [providers/base.py](src/providers/base.py) | `InferenceProvider` ABC, `ModelMetadata` dataclass | 183 |
| [providers/llamacpp.py](src/providers/llamacpp.py) | LlamaCpp-python provider for GGUF models with Metal | 580 |
| [providers/moondream.py](src/providers/moondream.py) | Moondream 2 VLM (HuggingFace) for image classification | 629 |
| [providers/deepseek_vl.py](src/providers/deepseek_vl.py) | DeepSeek VL2 Tiny provider (legacy, replaced by Moondream) | 637 |

#### Backends (Device Compatibility)

| Module | Purpose | Lines |
|--------|---------|-------|
| [backends/device_backend.py](src/providers/backends/device_backend.py) | `DeviceBackend` ABC + MPS/CUDA/CPU implementations | 268 |
| [backends/factory.py](src/providers/backends/factory.py) | Factory pattern for device-specific backends | 138 |
| [backends/siglip_attention_patch.py](src/providers/backends/siglip_attention_patch.py) | MPS attention fix for SigLIP models | ~100 |

#### Services

| Module | Purpose | Lines |
|--------|---------|-------|
| [services/model_manager.py](src/services/model_manager.py) | Model lifecycle, preset loading, role-based lookup | 478 |
| [services/cache.py](src/services/cache.py) | `InferenceCache` ABC, `PromptCache`, `HandoffCache` | 501 |
| [services/queue_manager.py](src/services/queue_manager.py) | Request queuing with FIFO/priority, concurrency control | 249 |

#### Core Utilities

| Module | Purpose | Lines |
|--------|---------|-------|
| [core/config.py](src/core/config.py) | Pydantic Settings with `INFERENCE_*` prefix | 238 |
| [core/constants.py](src/core/constants.py) | Container paths, port defaults | 51 |
| [core/exceptions.py](src/core/exceptions.py) | 15+ custom exceptions with Retriable/NonRetriable hierarchy | 554 |
| [core/logging.py](src/core/logging.py) | Structlog JSON logging, correlation ID support | 281 |

#### Orchestration

| Module | Purpose | Lines |
|--------|---------|-------|
| [orchestration/orchestrator.py](src/orchestration/orchestrator.py) | Dispatcher to orchestration modes | 171 |
| [orchestration/context.py](src/orchestration/context.py) | HandoffState, context budgets, error contamination | 460 |
| [orchestration/saga.py](src/orchestration/saga.py) | Saga compensation pattern for pipeline failures | 310 |
| [orchestration/modes/single.py](src/orchestration/modes/single.py) | Single model pass-through (only implemented mode) | 126 |

#### Schemas/Models (Pydantic)

| Module | Purpose | Lines |
|--------|---------|-------|
| [models/requests.py](src/models/requests.py) | `ChatCompletionRequest`, `Message`, `Tool`, `ToolCall` | 229 |
| [models/responses.py](src/models/responses.py) | `ChatCompletionResponse`, `ChatCompletionChunk`, `Usage` | 252 |

---

## 2. API Surface

### 2.1 Health Endpoints

| Method | Path | Response Schema | File | Line |
|--------|------|-----------------|------|------|
| GET | `/health` | `HealthResponse` | [health.py](src/api/routes/health.py) | L115 |
| GET | `/health/ready` | `ReadinessResponse` (200 or 503) | [health.py](src/api/routes/health.py) | L136 |

**HealthResponse Schema:**
```python
{
    "status": "ok",
    "service": "inference-service",
    "version": "0.1.0"
}
```

**ReadinessResponse Schema (200):**
```python
{
    "status": "ready",
    "loaded_models": ["phi-4", "qwen2.5-7b"],
    "config": "D3",
    "orchestration_mode": "single"
}
```

**ReadinessResponse Schema (503):**
```python
{
    "status": "not_ready",
    "loaded_models": [],
    "reason": "No models loaded"
}
```

---

### 2.2 Chat Completions Endpoint

| Method | Path | Request Schema | Response Schema | Streaming | File | Line |
|--------|------|----------------|-----------------|-----------|------|------|
| POST | `/v1/chat/completions` | `ChatCompletionRequest` | `ChatCompletionResponse` / SSE | Yes | [completions.py](src/api/routes/completions.py) | L107 |

**ChatCompletionRequest Schema:**
```python
{
    "model": "phi-4",                    # Required: model ID
    "messages": [                        # Required: conversation history
        {"role": "system", "content": "..."},
        {"role": "user", "content": "..."}
    ],
    "temperature": 0.7,                  # Optional: 0.0-2.0
    "max_tokens": 1024,                  # Optional: max generation
    "stream": false,                     # Optional: enable SSE streaming
    "top_p": 1.0,                        # Optional: nucleus sampling
    "stop": ["<|endoftext|>"],           # Optional: stop sequences
    "presence_penalty": 0.0,             # Optional: -2.0 to 2.0
    "frequency_penalty": 0.0,            # Optional: -2.0 to 2.0
    "tools": [...],                      # Optional: function calling
    "orchestration_mode": "single",      # Extension: orchestration mode
    "task_type": "code",                 # Extension: task routing
    "priority": 2                        # Extension: request priority
}
```

**ChatCompletionResponse Schema (Non-Streaming):**
```python
{
    "id": "chatcmpl-123",
    "object": "chat.completion",
    "created": 1737993600,
    "model": "phi-4",
    "choices": [{
        "index": 0,
        "message": {
            "role": "assistant",
            "content": "..."
        },
        "finish_reason": "stop"
    }],
    "usage": {
        "prompt_tokens": 100,
        "completion_tokens": 50,
        "total_tokens": 150
    },
    "orchestration": {                   # Extension field
        "mode": "single",
        "models_used": ["phi-4"],
        "total_inference_time_ms": 1234.5
    }
}
```

**Streaming (SSE Format):**
```
data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1737993600,"model":"phi-4","choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}

data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1737993600,"model":"phi-4","choices":[{"index":0,"delta":{"content":"Hello"},"finish_reason":null}]}

data: [DONE]
```

---

### 2.3 Models Endpoints

| Method | Path | Response Schema | File | Line |
|--------|------|-----------------|------|------|
| GET | `/v1/models` | `ModelsListResponse` | [models.py](src/api/routes/models.py) | L127 |
| POST | `/v1/models/{model_id}/load` | `ModelActionResponse` | [models.py](src/api/routes/models.py) | L168 |
| POST | `/v1/models/{model_id}/unload` | `ModelActionResponse` | [models.py](src/api/routes/models.py) | L217 |

**ModelsListResponse Schema:**
```python
{
    "data": [
        {
            "id": "phi-4",
            "status": "loaded",
            "memory_mb": 8400,
            "context_length": 4096,
            "roles": ["primary", "thinker", "coder"]
        },
        {
            "id": "qwen2.5-7b",
            "status": "available",
            "memory_mb": 4500,
            "context_length": 4500,
            "roles": ["coder", "primary", "fast"]
        }
    ],
    "config": "D3",
    "orchestration_mode": "single"
}
```

**ModelActionResponse Schema:**
```python
{
    "id": "phi-4",
    "status": "loaded",
    "message": "Model 'phi-4' loaded successfully"
}
```

---

### 2.4 Vision Endpoints

| Method | Path | Request Schema | Response Schema | File | Line |
|--------|------|----------------|-----------------|------|------|
| GET | `/api/v1/vision/health` | - | `VisionHealthResponse` | [vision.py](src/api/routes/vision.py) | L194 |
| POST | `/api/v1/vision/classify` | `ImageClassifyRequest` | `ImageClassifyResponse` | [vision.py](src/api/routes/vision.py) | L212 |
| POST | `/api/v1/vision/load` | - | `{status, model_id}` | [vision.py](src/api/routes/vision.py) | L307 |
| POST | `/api/v1/vision/unload` | - | `{status}` | [vision.py](src/api/routes/vision.py) | L329 |
| POST | `/api/v1/vision/classify/batch` | `BatchClassifyRequestAPI` | `BatchClassifyResponseAPI` | [vision.py](src/api/routes/vision.py) | L351 |

**ImageClassifyRequest Schema:**
```python
{
    "image_path": "/path/to/image.png",  # OR image_base64
    "image_base64": "base64string...",   # Alternative to path
    "prompt": "Custom prompt...",         # Optional
    "max_tokens": 512,                    # Optional: 1-4096
    "temperature": 0.1,                   # Optional: 0.0-2.0
    "resize_image": true                  # Optional: resize for speed
}
```

**ImageClassifyResponse Schema:**
```python
{
    "classification": "technical",        # "technical" or "discard"
    "raw_response": "technical",
    "confidence": 0.95,
    "model_id": "vikhyatk/moondream2",
    "usage": {
        "prompt_tokens": 256,
        "completion_tokens": 1,
        "total_tokens": 257
    }
}
```

**BatchClassifyResponseAPI Schema:**
```python
{
    "results": [
        {
            "image_path": "/path/to/image1.png",
            "classification": "technical",
            "raw_response": "technical",
            "confidence": 0.95,
            "error": null
        }
    ],
    "total_images": 10,
    "successful": 9,
    "failed": 1,
    "model_id": "vikhyatk/moondream2",
    "processing_time_seconds": 12.34,
    "images_per_second": 0.73
}
```

---

## 3. Interactions (What it Calls)

### 3.1 Internal Module Dependencies

```
main.py
├── api/routes/completions.py → services/model_manager.py
├── api/routes/health.py → services/model_manager.py
├── api/routes/models.py → services/model_manager.py
├── api/routes/vision.py → providers/moondream.py
├── core/config.py → core/constants.py
├── core/logging.py (singleton)
└── observability/tracing.py

services/model_manager.py
├── providers/llamacpp.py (for GGUF models)
├── core/config.py (get_settings)
└── config/models.yaml, presets.yaml (YAML configs)

providers/llamacpp.py
├── llama_cpp.Llama (external library)
├── models/requests.py
└── models/responses.py

providers/moondream.py
├── transformers.AutoModelForCausalLM
├── torch (MPS/CUDA/CPU)
└── PIL.Image

orchestration/orchestrator.py
├── orchestration/modes/single.py (only implemented)
└── providers/base.py

orchestration/saga.py
└── models/responses.py
```

### 3.2 Filesystem Paths

| Path | Purpose | Configured Via |
|------|---------|----------------|
| `/app/models` (container) | GGUF model files | `INFERENCE_MODELS_DIR` |
| `/app/config` (container) | models.yaml, presets.yaml | `INFERENCE_CONFIG_DIR` |
| `~/.cache/huggingface` | HuggingFace model cache (Moondream) | HF_HOME |

**Typical Native Paths:**
- Models: `/Users/kevintoles/POC/ai-models/models`
- Config: `/Users/kevintoles/POC/inference-service/config`

### 3.3 GPU/Metal Configuration

| Environment Variable | Default | Purpose |
|----------------------|---------|---------|
| `INFERENCE_GPU_LAYERS` | `-1` | GPU layers (-1=all Metal, 0=CPU, N=hybrid) |
| `INFERENCE_GPU_INDEX` | `0` | GPU device index for multi-GPU |
| `INFERENCE_BACKEND` | `llamacpp` | Inference backend |

**Per-Model GPU Control** (in models.yaml):
```yaml
phi-4:
  gpu_layers: 35  # 35/40 layers on GPU, 5 on CPU
deepseek-r1-7b:
  gpu_layers: -1  # All layers on Metal
phi-3-medium-128k:
  gpu_layers: 0   # CPU-only (8.6GB model)
```

### 3.4 Environment Variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `INFERENCE_PORT` | `8085` | HTTP server port |
| `INFERENCE_HOST` | `0.0.0.0` | Bind address |
| `INFERENCE_LOG_LEVEL` | `INFO` | Logging level |
| `INFERENCE_ENVIRONMENT` | `development` | dev/staging/production |
| `INFERENCE_MODELS_DIR` | `/app/models` | Model files directory |
| `INFERENCE_CONFIG_DIR` | `/app/config` | Config files directory |
| `INFERENCE_GPU_LAYERS` | `-1` | GPU offload layers |
| `INFERENCE_DEFAULT_PRESET` | `None` | Auto-load preset on startup |
| `INFERENCE_ORCHESTRATION_MODE` | `single` | Orchestration mode |
| `INFERENCE_VISION_MODEL_ID` | `vikhyatk/moondream2` | VLM model |
| `INFERENCE_VISION_MODEL_REVISION` | `2025-01-09` | VLM version |
| `INFERENCE_VISION_DEVICE` | `None` (auto) | VLM device |
| `OTLP_ENDPOINT` | `http://localhost:4317` | OpenTelemetry endpoint |
| `TRACING_ENABLED` | `true` | Enable distributed tracing |

---

## 4. Reverse References

### 4.1 Services That Call inference-service

| Service | File | Usage |
|---------|------|-------|
| **ai-agents** | [kitchen_brigade_executor.py](../ai-agents/src/protocols/kitchen_brigade_executor.py#L316) | Routes local model inference (qwen*, phi*, llama*) |
| **llm-gateway** | Provider routing | Routes to inference-service for local models |
| **context-management-service** | Proxy routing | CMS → inference-service for optimized requests |

### 4.2 Configuration References

| Repository | File | Reference |
|------------|------|-----------|
| **ai-models** | [README.md](../ai-models/README.md) | Model storage, points to inference-service config |
| **ai-platform-data** | [start_platform.sh](../ai-platform-data/start_platform.sh#L152) | Platform startup script |
| **ai-platform-data** | [tasks.json](.vscode/tasks.json) | VS Code tasks for starting service |

### 4.3 Import Graph (src/ modules)

```
main.py imports:
├── api.routes.completions.router
├── api.routes.health.router
├── api.routes.models.router
├── api.routes.vision.router
├── api.error_handlers.register_exception_handlers
├── core.config.get_settings
├── core.logging.configure_logging, get_logger
├── services.model_manager.get_model_manager
├── providers.moondream.MoondreamProvider
└── observability.setup_tracing, TracingMiddleware

api.routes.completions imports:
├── models.requests.ChatCompletionRequest
├── models.responses.ChatCompletionResponse
└── services.model_manager.ModelManager (TYPE_CHECKING)

services.model_manager imports:
├── providers.llamacpp.LlamaCppProvider
└── core.config.get_settings

providers.llamacpp imports:
├── models.requests.ChatCompletionRequest
├── models.responses.* (all response models)
└── providers.base.InferenceProvider, ModelMetadata
```

---

## 5. Issues / Observations

### 5.1 Missing Tests

| Module | Test File | Status |
|--------|-----------|--------|
| `routes/vision.py` | `test_vision.py` | ❌ **MISSING** |
| `providers/moondream.py` | `test_moondream.py` | ❌ **MISSING** |
| `providers/deepseek_vl.py` | `test_deepseek_vl.py` | ⚠️ Only MPS backend test |
| `orchestration/modes/critique.py` | - | ❌ **NOT IMPLEMENTED** |
| `orchestration/modes/debate.py` | - | ❌ **NOT IMPLEMENTED** |
| `orchestration/modes/ensemble.py` | - | ❌ **NOT IMPLEMENTED** |
| `orchestration/modes/pipeline.py` | - | ❌ **NOT IMPLEMENTED** |

**Existing Tests:**
- ✅ `tests/unit/api/routes/test_completions.py`
- ✅ `tests/unit/api/routes/test_health.py`
- ✅ `tests/unit/api/routes/test_models.py`
- ✅ `tests/unit/providers/test_llamacpp.py`
- ✅ `tests/unit/providers/test_base.py`
- ✅ `tests/unit/services/test_model_manager.py`
- ✅ `tests/unit/services/test_cache.py`
- ✅ `tests/unit/services/test_queue_manager.py`
- ✅ `tests/unit/orchestration/test_context.py`
- ✅ `tests/unit/orchestration/test_orchestrator.py`
- ✅ `tests/unit/orchestration/test_saga.py`
- ✅ `tests/integration/test_e2e.py`
- ✅ `tests/integration/test_streaming.py`

### 5.2 Memory Management Concerns

1. **VLM Model Loading** ([moondream.py#L288](src/providers/moondream.py#L288)):
   - Moondream (~4GB) loads on first inference request
   - No automatic unloading after idle period
   - Manual unload via `/api/v1/vision/unload` required

2. **GGUF Model Memory** ([model_manager.py#L94](src/services/model_manager.py#L94)):
   - Memory limit enforced (`memory_limit_gb=16.0` default)
   - No LRU eviction - manual unload required
   - Context length reduced from spec to fit 16GB Mac

3. **Concurrent Inference Lock** ([llamacpp.py#L179](src/providers/llamacpp.py#L179)):
   - `asyncio.Lock` serializes ALL inference requests per model
   - llama-cpp-python is NOT thread-safe (SIGABRT on concurrent decode)
   - Streaming holds lock for entire generation (no interleaving)

### 5.3 Model Loading Issues

1. **Graceful Degradation** ([model_manager.py#L294](src/services/model_manager.py#L294)):
   - If requested model not loaded, falls back to ANY loaded model
   - Logs warning but client may receive unexpected model response
   - Consider: return 404 instead of silent fallback?

2. **Preset Auto-Load** ([main.py#L78](src/main.py#L78)):
   - `INFERENCE_DEFAULT_PRESET` loads models at startup
   - Failure logs error but service starts without models
   - Readiness probe returns 503 until models loaded

3. **HuggingFace Download** ([moondream.py#L302](src/providers/moondream.py#L302)):
   - Moondream downloads from HF on first use (~2GB)
   - No progress indication during download
   - Download happens in thread pool (blocks first request)

### 5.4 Error Handling Gaps

1. **Exception Hierarchy** ([exceptions.py](src/core/exceptions.py)):
   - 15+ custom exceptions defined but not all used
   - `RetriableError` vs `NonRetriableError` distinction exists
   - Error response schema matches llm-gateway format ✅

2. **Streaming Error Recovery**:
   - If error occurs mid-stream, client receives `[DONE]` without error
   - No error chunk format defined (OpenAI spec doesn't define one)

3. **Vision Error Messages** ([vision.py#L279](src/api/routes/vision.py#L279)):
   - 503 returned when VLM loading fails with generic message
   - Consider adding more specific error codes

### 5.5 Orchestration Status

| Mode | Status | Implementation |
|------|--------|----------------|
| `single` | ✅ Implemented | [modes/single.py](src/orchestration/modes/single.py) |
| `critique` | ❌ Stub only | [modes/critique.py](src/orchestration/modes/critique.py) |
| `debate` | ❌ Stub only | [modes/debate.py](src/orchestration/modes/debate.py) |
| `ensemble` | ❌ Stub only | [modes/ensemble.py](src/orchestration/modes/ensemble.py) |
| `pipeline` | ❌ Stub only | [modes/pipeline.py](src/orchestration/modes/pipeline.py) |

**Note**: Attempting to use unimplemented modes raises `UnsupportedModeError`.

### 5.6 Configuration Issues

1. **Context Length Constraints** (models.yaml):
   - Many models have reduced context (e.g., phi-4: 4096 vs 16384 spec)
   - Necessary for 16GB Mac but undocumented to clients
   - `phi-3-medium-128k` limited to 8192 (CPU-only mode)

2. **Missing Validation**:
   - `models.yaml` model IDs must match directory names
   - No runtime validation of GGUF file integrity
   - Preset references can point to non-existent models

### 5.7 Observability Gaps

1. **Metrics** ([tracing.py](src/observability/tracing.py)):
   - OpenTelemetry tracing implemented ✅
   - Prometheus metrics endpoint NOT exposed
   - No model inference latency histograms

2. **Logging**:
   - Structured JSON logging via structlog ✅
   - Correlation ID propagation ✅
   - Missing: inference token counts per request in logs

---

## 6. Configuration Files

### 6.1 models.yaml

Location: `config/models.yaml`

Defines 10+ models with:
- Model ID, name, description
- File path (relative to INFERENCE_MODELS_DIR)
- Size (GB), context length, quantization
- Roles (primary, thinker, coder, fast, longctx)
- GPU layers (-1=all GPU, 0=CPU, N=hybrid)

### 6.2 presets.yaml

Location: `config/presets.yaml`

Defines 33+ configuration presets:
- **Single (S1-S10)**: 1 model configurations
- **Dual (D1-D15)**: 2 model combinations
- **Triple (T1-T13)**: 3 model combinations
- **Quad (Q1-Q7)**: 4 model combinations
- **Penta (P1-P6)**: 5 model combinations

Each preset specifies:
- Models to load
- Total memory requirement
- Orchestration mode
- Target hardware (Mac 16GB, Server 24GB+, etc.)

---

## 7. Summary Statistics

| Metric | Value |
|--------|-------|
| Total Python files | 37 |
| Total lines of code | ~7,500 |
| API Endpoints | 11 |
| Custom Exceptions | 15+ |
| Pydantic Models | 20+ |
| Test files | 20+ |
| Model presets | 33+ |
| Supported models | 10+ |

---

## 8. Quick Reference

### Start Service (Native/Hybrid)
```bash
cd /Users/kevintoles/POC/inference-service
source .venv/bin/activate
INFERENCE_GPU_LAYERS=-1 \
INFERENCE_MODELS_DIR=/Users/kevintoles/POC/ai-models/models \
INFERENCE_CONFIG_DIR=/Users/kevintoles/POC/inference-service/config \
uvicorn src.main:app --host 0.0.0.0 --port 8085
```

### Health Check
```bash
curl http://localhost:8085/health
curl http://localhost:8085/health/ready
```

### Load Model
```bash
curl -X POST http://localhost:8085/v1/models/phi-4/load
```

### Chat Completion
```bash
curl -X POST http://localhost:8085/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "phi-4",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

### Vision Classification
```bash
curl -X POST http://localhost:8085/api/v1/vision/classify \
  -H "Content-Type: application/json" \
  -d '{
    "image_path": "/path/to/image.png"
  }'
```
