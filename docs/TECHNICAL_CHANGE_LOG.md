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
