# GPU Allocation Semantics

**Version**: 1.0.0  
**Created**: January 18, 2026  
**Status**: ACTIVE  
**WBS Reference**: WBS_B5_B1_REMAINING_WORK.md - GPU-1 through GPU-4

---

## Overview

This document defines GPU resource allocation semantics for the inference-service, addressing Architecture Roundtable finding B1 (GPU Allocation Semantics Undefined).

**Scope:**
- GPU scheduling authority
- OOM handling and recovery
- Driver reset behavior
- Multi-GPU selection rules

---

## 1. GPU Scheduling Authority

### Single Authority: inference-service

The **inference-service** is the single authority for GPU allocation within the Kitchen Brigade platform.

```
┌─────────────────────────────────────────────────────────────────┐
│                    Request Flow with GPU                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   Client ──▶ llm-gateway ──▶ inference-service ──▶ GPU           │
│                  │                  │                            │
│                  │                  ├─ Owns GPU allocation       │
│                  │                  ├─ Enforces memory limits    │
│                  │                  └─ Manages model loading     │
│                  │                                               │
│                  └── Circuit breaker protection                  │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

**Key Points:**
- No other service directly allocates GPU resources
- llm-gateway delegates to inference-service for local model inference
- llm-gateway handles failover to cloud providers if inference-service fails

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `INFERENCE_GPU_LAYERS` | `-1` | Layers on GPU (-1=all, 0=CPU-only, N=hybrid) |
| `INFERENCE_GPU_INDEX` | `0` | GPU device index for multi-GPU systems |
| `INFERENCE_MEMORY_LIMIT_GB` | `16.0` | Maximum memory allowed for models |

---

## 2. OOM Handling: Fail-Fast Strategy

### Strategy: Crash → Circuit Breaker → Fallback

The inference-service uses a **fail-fast** strategy for OOM conditions rather than degrading to CPU.

```
┌─────────────────────────────────────────────────────────────────┐
│                    OOM Recovery Flow                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  1. OOM Detected                                                 │
│     ├─ Model load fails with MemoryLimitExceededError            │
│     └─ System OOM kills process (hard failure)                   │
│                                                                  │
│  2. Circuit Breaker Trips (llm-gateway)                          │
│     ├─ State: CLOSED → OPEN                                      │
│     ├─ Failure threshold: 5 consecutive failures                 │
│     └─ Reset timeout: 30 seconds                                 │
│                                                                  │
│  3. Fallback Chain Executes                                      │
│     ├─ Tier 1: Other local models (smaller context)              │
│     ├─ Tier 2: Cloud provider (Anthropic/OpenAI/Google)          │
│     └─ Tier 3: Return 503 Service Unavailable                    │
│                                                                  │
│  4. Recovery Testing (Half-Open State)                           │
│     ├─ After 30s, allow 1 probe request                          │
│     ├─ Success → CLOSED, resume local inference                  │
│     └─ Failure → OPEN, continue fallback                         │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Pre-Loading Memory Check

The inference-service prevents OOM before it happens:

```python
# src/services/model_manager.py lines 225-233
if self.current_memory_gb + model_size > self.memory_limit_gb:
    raise MemoryLimitExceededError(
        f"Loading '{model_id}' ({model_size:.1f}GB) would exceed "
        f"memory limit ({self.current_memory_gb:.1f}GB + "
        f"{model_size:.1f}GB > {self.memory_limit_gb:.1f}GB)"
    )
```

### Why Fail-Fast Instead of Degrade-to-CPU?

| Approach | Pros | Cons |
|----------|------|------|
| **Fail-Fast (Chosen)** | Predictable latency, clear error signals | Requires fallback chain |
| Degrade-to-CPU | Service stays "up" | 10-100x slower, unpredictable latency |
| Reroute | Distributes load | Requires multi-instance coordination |

**Decision:** Fail-fast with circuit breaker is preferred because:
1. CPU inference is too slow for production use (10-100x slower)
2. Cloud fallback provides better UX than degraded local performance
3. Circuit breaker prevents cascading failures

---

## 3. Driver Reset Behavior

### Detection Mechanisms

GPU driver resets can occur due to:
- Timeout Detection and Recovery (TDR) on Windows
- GPU hang/watchdog on Linux/macOS
- CUDA driver crash
- Metal shader compilation failure

**Symptoms in llama-cpp-python:**
```
RuntimeError: Metal initialization failed
RuntimeError: CUDA error: device-side assert triggered
RuntimeError: cuBLAS initialization failed
```

### Recovery Strategy

```
┌─────────────────────────────────────────────────────────────────┐
│                Driver Reset Recovery Flow                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  1. Driver Reset Detected                                        │
│     ├─ LlamaCppModelLoadError raised                             │
│     └─ inference-service logs error                              │
│                                                                  │
│  2. Model State Cleared                                          │
│     ├─ _model = None                                             │
│     ├─ _is_loaded = False                                        │
│     └─ Memory tracking decremented                               │
│                                                                  │
│  3. Circuit Breaker Trips                                        │
│     ├─ llm-gateway marks inference-service unhealthy             │
│     └─ Requests fail fast                                        │
│                                                                  │
│  4. Recovery Options                                             │
│     ├─ Auto-restart: Container orchestrator restarts service     │
│     ├─ Manual restart: Operator intervention required            │
│     └─ Fallback: Cloud providers handle traffic during recovery  │
│                                                                  │
│  5. Circuit Breaker Probes Recovery                              │
│     ├─ After reset_timeout, send probe request                   │
│     ├─ Model reload attempted on first inference                 │
│     └─ Success clears circuit breaker                            │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Container Restart Policy

For Docker deployments, use restart policy:

```yaml
# docker-compose.yml
inference-service:
  restart: unless-stopped
  deploy:
    resources:
      reservations:
        devices:
          - driver: nvidia
            count: 1
            capabilities: [gpu]
```

**Note:** On macOS with Metal, driver resets are rare. The Metal API handles shader compilation errors gracefully.

---

## 4. Multi-GPU Selection Rules

### Platform Support

| Platform | GPU Type | Selection Mechanism |
|----------|----------|-------------------|
| **macOS** | Metal | Automatic (uses first discrete GPU) |
| **Linux** | NVIDIA CUDA | `CUDA_VISIBLE_DEVICES` or `INFERENCE_GPU_INDEX` |
| **Linux** | AMD ROCm | `HIP_VISIBLE_DEVICES` |
| **Windows** | NVIDIA CUDA | `CUDA_VISIBLE_DEVICES` |

### Selection Priority

1. **Environment Variable** (`INFERENCE_GPU_INDEX`) - explicit selection
2. **llama-cpp Default** - first available GPU
3. **CPU Fallback** - if GPU unavailable or `gpu_layers=0`

### macOS Metal (Apple Silicon)

Apple Silicon Macs have unified memory, so GPU selection is simplified:

```python
# src/providers/llamacpp.py
if n_gpu_layers == -1 and platform.system() == "Darwin":
    # Metal acceleration for all layers
    n_gpu_layers = -1
```

**Key Points:**
- Metal automatically uses the available GPU
- No multi-GPU selection needed (single unified GPU)
- `n_gpu_layers=-1` enables full Metal acceleration

### Linux/Windows CUDA Multi-GPU

For multi-GPU NVIDIA systems:

```bash
# Method 1: System-wide (set before starting service)
export CUDA_VISIBLE_DEVICES=0  # Use first GPU
export CUDA_VISIBLE_DEVICES=1  # Use second GPU
export CUDA_VISIBLE_DEVICES=0,1  # Use both (model parallel)

# Method 2: inference-service config
export INFERENCE_GPU_INDEX=0  # Select by index
```

**Implementation:**

```python
# src/core/config.py
gpu_index: int = Field(
    default=0,
    ge=0,
    description="GPU device index for multi-GPU systems (CUDA/ROCm)"
)

# Applied in model loading
import os
if settings.gpu_index is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(settings.gpu_index)
```

### Per-Model GPU Layers

Models can override the global `gpu_layers` setting:

```yaml
# config/models.yaml
models:
  phi-4:
    path: "phi-4-Q4_K_M.gguf"
    context_length: 16384
    gpu_layers: -1  # All layers on GPU
    
  codellama-7b:
    path: "codellama-7b.Q4_K_M.gguf"
    context_length: 8192
    gpu_layers: 20  # Only 20 layers on GPU (hybrid)
```

---

## 5. Configuration Reference

### Complete Environment Variables

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `INFERENCE_GPU_LAYERS` | `int` | `-1` | Layers to offload to GPU (-1=all) |
| `INFERENCE_GPU_INDEX` | `int` | `0` | GPU device index (multi-GPU) |
| `INFERENCE_MEMORY_LIMIT_GB` | `float` | `16.0` | Max memory for loaded models |
| `INFERENCE_BACKEND` | `str` | `llamacpp` | Inference backend |

### GPU Layer Values

| Value | Meaning | Use Case |
|-------|---------|----------|
| `-1` | All layers on GPU | Maximum performance |
| `0` | CPU only | No GPU available |
| `1-N` | N layers on GPU | Hybrid (large models) |

### Memory Calculation

```
Required Memory ≈ Model Size + KV Cache + Activations

For a 7B model (Q4_K_M quantization):
- Model: ~4GB
- KV Cache (8K context): ~0.5GB
- Activations: ~0.5GB
- Total: ~5GB
```

---

## 6. Troubleshooting

### Common Issues

| Symptom | Cause | Solution |
|---------|-------|----------|
| `MemoryLimitExceededError` | Model too large | Reduce `context_length` or use smaller model |
| `Metal initialization failed` | macOS GPU busy | Restart inference-service |
| `CUDA out of memory` | GPU VRAM exhausted | Reduce `gpu_layers` or use smaller batch |
| Circuit breaker stuck OPEN | Repeated failures | Check logs, restart service |

### Diagnostic Commands

```bash
# Check GPU status (NVIDIA)
nvidia-smi

# Check GPU status (macOS)
system_profiler SPDisplaysDataType

# Check inference-service health
curl http://localhost:8085/health

# Check loaded models
curl http://localhost:8085/v1/models
```

---

## 7. Future Work

### Planned Improvements

1. **HardwareAllocator Interface** - Abstraction for GPU/CPU allocation (see HARDWARE_ABSTRACTION.md)
2. **Dynamic GPU Switching** - Hot-swap between GPUs during runtime
3. **Multi-Instance Coordination** - Distribute load across inference-service replicas
4. **GPU Memory Pooling** - Share GPU memory across models

---

## Citations

| # | Source | Section |
|---|--------|---------|
| [^1] | ARCHITECTURE_ROUNDTABLE_FINDINGS.md | §B1: GPU Allocation Semantics |
| [^2] | src/services/model_manager.py | Lines 225-233: Memory limit check |
| [^3] | src/providers/llamacpp.py | Lines 230-245: Metal acceleration |
| [^4] | src/core/config.py | GPU configuration fields |
| [^5] | llm-gateway/src/resilience/circuit_breaker_state_machine.py | Circuit breaker states |
| [^6] | Building Reactive Microservices (Escoffier) | Ch.6: Circuit Breaker Pattern |

