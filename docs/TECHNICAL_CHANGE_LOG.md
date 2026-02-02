# inference-service: Technical Change Log

**Purpose**: Documents architectural decisions, conflict resolutions, and significant changes to the inference service.

---

## Changelog

### 2026-01-31: Vision/VLM Capabilities Added (CL-004)

**Summary**: Added vision/VLM capabilities supporting DeepSeek-VL2 and Moondream models for image analysis.

**Background**:
- Platform needed multimodal capabilities for code screenshot analysis
- Vision-language models enable diagram parsing and UI screenshot understanding
- Metal GPU acceleration critical for VLM inference performance

**Changes Made**:

| File | Change |
|------|--------|
| `src/providers/deepseek_vl.py` | New provider for DeepSeek-VL2-Tiny |
| `src/providers/moondream.py` | New provider for Moondream-2B |
| `src/api/routes/vision.py` | New vision endpoints |
| `src/services/vision_service.py` | Vision orchestration layer |
| `config/models.yaml` | Added vision model configurations |
| `docs/ARCHITECTURE.md` | Added `/v1/vision/*` endpoint documentation |

**New API Endpoints**:

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/v1/vision/analyze` | POST | Analyze image with prompt |
| `/v1/vision/describe` | POST | Generate image description |
| `/v1/vision/extract` | POST | Extract structured data from image |
| `/v1/vision/compare` | POST | Compare two images |
| `/v1/vision/models` | GET | List available vision models |

**Architecture Alignment**:
- ✅ Kitchen Brigade: Sous Chef Worker expanded capabilities
- ✅ Multimodal: Enables code diagram and screenshot analysis
- ✅ Metal GPU: VLM inference optimized for Apple Silicon

---

### 2026-01-28: OpenTelemetry Distributed Tracing (CL-003)

**Summary**: Added OpenTelemetry distributed tracing for observability (OBS-11).

**Background**:
- Platform needed end-to-end request tracing across services
- Debugging multi-service requests required correlated traces
- OTEL standard enables integration with Jaeger/Tempo/etc.

**Changes Made**:

| File | Change |
|------|--------|
| `src/observability/tracing.py` | New tracing module |
| `src/main.py` | OTEL middleware integration |
| `requirements.txt` | Added `opentelemetry-*` dependencies |
| `docker-compose.yml` | Added OTEL environment variables |

**Environment Variables**:
```bash
OTEL_SERVICE_NAME=inference-service
OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317
OTEL_TRACES_EXPORTER=otlp
```

**Architecture Alignment**:
- ✅ Observability: Traces flow through Kitchen Brigade services
- ✅ WBS OBS-11: Distributed tracing requirement satisfied
- ✅ Integration: Compatible with platform-wide Jaeger collector

---

### 2025-12-31: Model Configuration Ownership Established (CL-001)

**Summary**: Established inference-service as the **single source of truth** for all model configuration. Deleted duplicate config files from ai-models repo.

**Background**:
- `models.yaml` existed in both `ai-models/config/` and `inference-service/config/`
- Configuration drift occurred (different gpu_layers values, path formats)
- Violated microservices autonomy principle

**Conflict Resolution**:

| Priority | Document | Guidance Applied |
|----------|----------|------------------|
| 1 | GUIDELINES_AI_Engineering | ML systems require clear ownership |
| 2 | AI_CODING_PLATFORM_ARCHITECTURE | Kitchen Brigade: each service owns its domain |
| 3 | llm-gateway ARCHITECTURE | No responsibility duplication |
| 4 | Building Microservices | Service autonomy - deploy independently |
| 5 | CODING_PATTERNS_ANALYSIS | Single source of truth |

**Changes Made**:

| File | Action |
|------|--------|
| `ai-models/config/models.yaml` | Deleted |
| `ai-models/config/configs.yaml` | Deleted |
| `ai-models/README.md` | Updated (storage-only role) |
| `inference-service/docs/ARCHITECTURE.md` | Updated (ownership clarified) |

**Configuration Ownership**:

```
inference-service/config/
├── models.yaml      # Model metadata, gpu_layers, context_length, roles
└── presets.yaml     # 33 configuration presets (S1-S8, D1-D15, etc.)

ai-models/
└── models/          # .gguf files only (NO config files)
```

**Environment Variables**:
```bash
INFERENCE_MODELS_DIR=/path/to/ai-models/models    # Points to storage
INFERENCE_CONFIG_DIR=/path/to/inference-service/config  # Owned by this service
```

**Rationale**:
1. inference-service is the Kitchen Brigade "Sous Chef Worker" - it operates models
2. ai-models is storage (like a pantry) - no intelligence, no config
3. Microservices principle: "Can you deploy without changing anything else?"

---

### 2026-01-01: Graceful Model Degradation & Docker Restart Policy (CL-002)

**Summary**: Implemented graceful model degradation in ModelManager - if requested model isn't loaded, system uses any available model. Changed Docker restart policy to prevent auto-start interfering with native Metal GPU execution.

**Background**:
- Clients were hardcoding model preferences (e.g., `qwen2.5-7b`)
- When only `deepseek-r1-7b` was loaded (S2 preset), requests failed
- Docker Desktop was auto-starting the container, defeating Metal GPU acceleration
- Native execution required for Apple Silicon GPU utilization

**Changes Made**:

| File | Change |
|------|--------|
| `src/services/model_manager.py` | Added graceful degradation to `get_provider()` |
| `docker/docker-compose.yml` | Changed `restart: unless-stopped` to `restart: "no"` |

**Graceful Degradation Logic in ModelManager.get_provider()**:

```python
def get_provider(self, model_name: str | None = None) -> BaseProvider:
    """Get provider for requested model, with graceful degradation."""
    loaded_models = list(self._providers.keys())
    
    if not loaded_models:
        raise RuntimeError("No models loaded. Load a model first.")
    
    if model_name is None or model_name in loaded_models:
        # Use requested model or first loaded
        target = model_name or loaded_models[0]
        return self._providers[target]
    
    # Graceful degradation: use any loaded model
    fallback = loaded_models[0]
    logger.warning(f"Model '{model_name}' not loaded, using '{fallback}'")
    return self._providers[fallback]
```

**Docker Restart Policy Rationale**:

| Setting | Previous | New | Reason |
|---------|----------|-----|--------|
| `restart` | `unless-stopped` | `"no"` | Prevent Docker Desktop auto-start |

**Native Execution Preference**:
```bash
# Start native (Metal GPU)
cd ~/POC/inference-service
./start_inference.sh --preset S2  # deepseek-only, 4.7GB

# Docker only for testing/CI
docker compose up -d  # No auto-restart on reboot
```

**Architecture Alignment**:
- ✅ Kitchen Brigade: Sous Chef Worker adapts to available resources
- ✅ Supports WBS-KB10: Summarization pipeline works with any loaded model
- ✅ Metal GPU: Native execution required for Apple Silicon acceleration

**Conflict Resolution Priority**:
| Priority | Document | Guidance Applied |
|----------|----------|------------------|
| 1 | GUIDELINES_AI_Engineering | "Fail gracefully, degrade not crash" |
| 2 | AI_CODING_PLATFORM_ARCHITECTURE | Services adapt to runtime constraints |
| 3 | Building Microservices | Resilience over rigid expectations |

---
