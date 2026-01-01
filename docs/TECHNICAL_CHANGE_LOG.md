# inference-service: Technical Change Log

**Purpose**: Documents architectural decisions, conflict resolutions, and significant changes to the inference service.

---

## Changelog

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
