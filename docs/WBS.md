# inference-service Work Breakdown Structure (WBS)

> **Version:** 1.0.0  
> **Created:** 2025-12-27  
> **Status:** Planning Phase  
> **Reference:** [ARCHITECTURE.md](ARCHITECTURE.md)

## Overview

This WBS defines the implementation tasks, exit criteria, and acceptance criteria for the inference-service. Each WBS block is self-contained - when implementation and exit criteria are satisfied, the acceptance criteria is automatically satisfied.

**TDD Approach:** All implementation follows RED → GREEN → REFACTOR cycle.

---

## WBS Summary

| Block | Name | Dependencies | Est. Effort |
|-------|------|--------------|-------------|
| WBS-INF1 | Repository Scaffolding | None | 2 hours |
| WBS-INF2 | Core Infrastructure | WBS-INF1 | 4 hours |
| WBS-INF3 | Pydantic Models | WBS-INF2 | 4 hours |
| WBS-INF4 | Provider Abstraction | WBS-INF3 | 6 hours |
| WBS-INF5 | LlamaCpp Provider | WBS-INF4 | 8 hours |
| WBS-INF6 | Model Manager Service | WBS-INF5 | 6 hours |
| WBS-INF7 | API Routes - Health | WBS-INF6 | 2 hours |
| WBS-INF8 | API Routes - Models | WBS-INF6 | 4 hours |
| WBS-INF9 | API Routes - Completions | WBS-INF6 | 8 hours |
| WBS-INF10 | Context Management | WBS-INF9 | 8 hours |
| WBS-INF11 | Orchestration - Single Mode | WBS-INF10 | 4 hours |
| WBS-INF12 | Orchestration - Critique Mode | WBS-INF11 | 6 hours |
| WBS-INF13 | Orchestration - Pipeline Mode | WBS-INF11 | 6 hours |
| WBS-INF14 | Orchestration - Debate Mode | WBS-INF11 | 6 hours |
| WBS-INF15 | Orchestration - Ensemble Mode | WBS-INF11 | 6 hours |
| WBS-INF16 | Caching Infrastructure | WBS-INF10 | 6 hours |
| WBS-INF17 | Queue Manager | WBS-INF9 | 4 hours |
| WBS-INF18 | Error Handling | WBS-INF9 | 4 hours |
| WBS-INF19 | Docker & CI/CD | WBS-INF9 | 4 hours |
| WBS-INF20 | Anti-Pattern Compliance | All prior WBS | 4 hours |
| WBS-INF21 | Integration Testing | All prior WBS | 8 hours |

**Total Estimated Effort:** ~100 hours

---

## WBS-INF1: Repository Scaffolding

**Dependencies:** None  
**Reference:** ARCHITECTURE.md → Folder Structure

### Acceptance Criteria

| ID | Requirement |
|----|-------------|
| AC-1.1 | Folder structure matches ARCHITECTURE.md specification |
| AC-1.2 | All `__init__.py` files created for Python packages |
| AC-1.3 | pyproject.toml configured with project metadata |
| AC-1.4 | .gitignore excludes appropriate files |
| AC-1.5 | .env.example contains all environment variables |

### WBS Tasks

| ID | Task | AC | File(s) |
|----|------|-----|---------|
| INF1.1 | Create src/ directory structure | AC-1.1 | `src/`, `src/api/`, `src/api/routes/`, `src/core/`, `src/models/`, `src/providers/`, `src/orchestration/`, `src/orchestration/modes/`, `src/services/` |
| INF1.2 | Create tests/ directory structure | AC-1.1 | `tests/`, `tests/unit/`, `tests/unit/providers/`, `tests/unit/orchestration/`, `tests/unit/services/`, `tests/integration/` |
| INF1.3 | Create config/ directory | AC-1.1 | `config/` |
| INF1.4 | Create docker/ directory | AC-1.1 | `docker/` |
| INF1.5 | Create all __init__.py files | AC-1.2 | All package directories |
| INF1.6 | Create pyproject.toml | AC-1.3 | `pyproject.toml` |
| INF1.7 | Create requirements.txt | AC-1.3 | `requirements.txt` |
| INF1.8 | Create .gitignore | AC-1.4 | `.gitignore` |
| INF1.9 | Create .env.example | AC-1.5 | `.env.example` |
| INF1.10 | Create README.md | AC-1.1 | `README.md` |
| INF1.11 | Create conftest.py | AC-1.1 | `tests/conftest.py` |

### Exit Criteria

- [ ] `tree src/` shows all directories from ARCHITECTURE.md
- [ ] `python -c "import src"` succeeds without errors
- [ ] `pip install -e .` succeeds
- [ ] `.env.example` contains all 25+ environment variables from Configuration Reference

---

## WBS-INF2: Core Infrastructure

**Dependencies:** WBS-INF1  
**Reference:** ARCHITECTURE.md → Configuration Reference

### Acceptance Criteria

| ID | Requirement |
|----|-------------|
| AC-2.1 | Pydantic Settings loads all environment variables |
| AC-2.2 | Structured logging with JSON format |
| AC-2.3 | FastAPI app initializes without errors |
| AC-2.4 | Configuration validates on startup |

### WBS Tasks

| ID | Task | AC | File(s) |
|----|------|-----|---------|
| INF2.1 | RED: Write config tests | AC-2.1 | `tests/unit/core/test_config.py` |
| INF2.2 | GREEN: Implement Settings class | AC-2.1 | `src/core/config.py` |
| INF2.3 | RED: Write logging tests | AC-2.2 | `tests/unit/core/test_logging.py` |
| INF2.4 | GREEN: Implement structured logging | AC-2.2 | `src/core/logging.py` |
| INF2.5 | RED: Write main app tests | AC-2.3 | `tests/unit/test_main.py` |
| INF2.6 | GREEN: Implement FastAPI app entry | AC-2.3, AC-2.4 | `src/main.py` |
| INF2.7 | REFACTOR: Extract common utilities | AC-2.1-2.4 | As needed |

### Exit Criteria

- [ ] `pytest tests/unit/core/` passes with 100% coverage on config.py, logging.py
- [ ] `INFERENCE_PORT=9999 python -c "from src.core.config import settings; assert settings.port == 9999"`
- [ ] Log output is valid JSON with timestamp, level, message
- [ ] Invalid config raises `ValidationError` with clear message

---

## WBS-INF3: Pydantic Models

**Dependencies:** WBS-INF2  
**Reference:** ARCHITECTURE.md → API Contract

### Acceptance Criteria

| ID | Requirement |
|----|-------------|
| AC-3.1 | ChatCompletionRequest matches OpenAI spec + extensions |
| AC-3.2 | ChatCompletionResponse matches OpenAI spec + orchestration |
| AC-3.3 | Streaming chunk models support SSE format |
| AC-3.4 | All models have JSON schema export |
| AC-3.5 | Models validate input with clear error messages |

### WBS Tasks

| ID | Task | AC | File(s) |
|----|------|-----|---------|
| INF3.1 | RED: Write request model tests | AC-3.1, AC-3.5 | `tests/unit/models/test_requests.py` |
| INF3.2 | GREEN: Implement ChatCompletionRequest | AC-3.1 | `src/models/requests.py` |
| INF3.3 | Implement Message, Tool, ToolChoice | AC-3.1 | `src/models/requests.py` |
| INF3.4 | RED: Write response model tests | AC-3.2, AC-3.4 | `tests/unit/models/test_responses.py` |
| INF3.5 | GREEN: Implement ChatCompletionResponse | AC-3.2 | `src/models/responses.py` |
| INF3.6 | Implement Choice, Usage, OrchestrationMetadata | AC-3.2 | `src/models/responses.py` |
| INF3.7 | RED: Write chunk model tests | AC-3.3 | `tests/unit/models/test_responses.py` |
| INF3.8 | GREEN: Implement ChatCompletionChunk, ChunkDelta | AC-3.3 | `src/models/responses.py` |
| INF3.9 | Implement ModelInfo for /v1/models | AC-3.4 | `src/models/responses.py` |
| INF3.10 | REFACTOR: Add model_json_schema() exports | AC-3.4 | All model files |

### Exit Criteria

- [ ] `pytest tests/unit/models/` passes with 100% coverage
- [ ] Request JSON from ARCHITECTURE.md parses without error
- [ ] Response JSON from ARCHITECTURE.md validates against schema
- [ ] `ChatCompletionRequest.model_json_schema()` returns valid JSON Schema

---

## WBS-INF4: Provider Abstraction

**Dependencies:** WBS-INF3  
**Reference:** ARCHITECTURE.md → Folder Structure (providers/)

### Acceptance Criteria

| ID | Requirement |
|----|-------------|
| AC-4.1 | InferenceProvider ABC defines complete interface |
| AC-4.2 | Provider supports both sync and streaming generation |
| AC-4.3 | Provider exposes model metadata (context length, roles) |
| AC-4.4 | Provider interface is testable with mock implementation |

### WBS Tasks

| ID | Task | AC | File(s) |
|----|------|-----|---------|
| INF4.1 | RED: Write provider interface tests | AC-4.1, AC-4.4 | `tests/unit/providers/test_base.py` |
| INF4.2 | GREEN: Implement InferenceProvider ABC | AC-4.1 | `src/providers/base.py` |
| INF4.3 | Define generate() abstract method | AC-4.2 | `src/providers/base.py` |
| INF4.4 | Define stream() abstract method | AC-4.2 | `src/providers/base.py` |
| INF4.5 | Define model_info property | AC-4.3 | `src/providers/base.py` |
| INF4.6 | Define tokenize() and count_tokens() | AC-4.1 | `src/providers/base.py` |
| INF4.7 | GREEN: Implement MockProvider for testing | AC-4.4 | `tests/unit/providers/mock_provider.py` |
| INF4.8 | REFACTOR: Add Protocol type hints | AC-4.1 | `src/providers/base.py` |

### Exit Criteria

- [ ] `pytest tests/unit/providers/test_base.py` passes
- [ ] MockProvider passes all interface tests
- [ ] `mypy src/providers/base.py` reports 0 errors
- [ ] ABC prevents instantiation without implementing all methods

---

## WBS-INF5: LlamaCpp Provider

**Dependencies:** WBS-INF4  
**Reference:** ARCHITECTURE.md → Model Configuration

### Acceptance Criteria

| ID | Requirement |
|----|-------------|
| AC-5.1 | LlamaCppProvider loads GGUF models from disk |
| AC-5.2 | Provider generates completions using llama-cpp-python |
| AC-5.3 | Provider supports streaming with proper SSE format |
| AC-5.4 | Provider correctly reports token usage |
| AC-5.5 | Provider uses Metal acceleration on Mac |
| AC-5.6 | Provider handles model loading/unloading |

### WBS Tasks

| ID | Task | AC | File(s) |
|----|------|-----|---------|
| INF5.1 | RED: Write provider initialization tests | AC-5.1, AC-5.5 | `tests/unit/providers/test_llamacpp.py` |
| INF5.2 | GREEN: Implement __init__ with model loading | AC-5.1, AC-5.5 | `src/providers/llamacpp.py` |
| INF5.3 | RED: Write generate() tests | AC-5.2, AC-5.4 | `tests/unit/providers/test_llamacpp.py` |
| INF5.4 | GREEN: Implement generate() method | AC-5.2, AC-5.4 | `src/providers/llamacpp.py` |
| INF5.5 | RED: Write stream() tests | AC-5.3 | `tests/unit/providers/test_llamacpp.py` |
| INF5.6 | GREEN: Implement stream() method | AC-5.3 | `src/providers/llamacpp.py` |
| INF5.7 | RED: Write tokenize tests | AC-5.4 | `tests/unit/providers/test_llamacpp.py` |
| INF5.8 | GREEN: Implement tokenize() and count_tokens() | AC-5.4 | `src/providers/llamacpp.py` |
| INF5.9 | RED: Write load/unload tests | AC-5.6 | `tests/unit/providers/test_llamacpp.py` |
| INF5.10 | GREEN: Implement load() and unload() | AC-5.6 | `src/providers/llamacpp.py` |
| INF5.11 | REFACTOR: Add context manager support | AC-5.6 | `src/providers/llamacpp.py` |

### Exit Criteria

- [ ] `pytest tests/unit/providers/test_llamacpp.py` passes (may skip if no model available)
- [ ] Integration test with llama-3.2-3b generates valid response
- [ ] Streaming produces valid SSE chunks ending with `[DONE]`
- [ ] `n_gpu_layers=-1` activates Metal on Mac

---

## WBS-INF6: Model Manager Service

**Dependencies:** WBS-INF5  
**Reference:** ARCHITECTURE.md → Concurrency & Scaling

### Acceptance Criteria

| ID | Requirement |
|----|-------------|
| AC-6.1 | ModelManager loads models from config preset |
| AC-6.2 | ModelManager tracks loaded/available models |
| AC-6.3 | ModelManager enforces memory limits |
| AC-6.4 | ModelManager supports concurrent model access |
| AC-6.5 | ModelManager provides model by role lookup |

### WBS Tasks

| ID | Task | AC | File(s) |
|----|------|-----|---------|
| INF6.1 | RED: Write model registry tests | AC-6.1, AC-6.2 | `tests/unit/services/test_model_manager.py` |
| INF6.2 | GREEN: Implement model registry | AC-6.1, AC-6.2 | `src/services/model_manager.py` |
| INF6.3 | RED: Write memory management tests | AC-6.3 | `tests/unit/services/test_model_manager.py` |
| INF6.4 | GREEN: Implement memory tracking | AC-6.3 | `src/services/model_manager.py` |
| INF6.5 | RED: Write concurrent access tests | AC-6.4 | `tests/unit/services/test_model_manager.py` |
| INF6.6 | GREEN: Implement asyncio locks for model access | AC-6.4 | `src/services/model_manager.py` |
| INF6.7 | RED: Write role-based lookup tests | AC-6.5 | `tests/unit/services/test_model_manager.py` |
| INF6.8 | GREEN: Implement get_model_by_role() | AC-6.5 | `src/services/model_manager.py` |
| INF6.9 | Implement config preset loading | AC-6.1 | `src/services/model_manager.py` |
| INF6.10 | Create model_configs.yaml | AC-6.1 | `config/model_configs.yaml` |

### Exit Criteria

- [ ] `pytest tests/unit/services/test_model_manager.py` passes with 100% coverage
- [ ] Config preset `D3` loads phi-4 and deepseek-r1-7b
- [ ] Memory check prevents loading models exceeding 16GB
- [ ] `get_model_by_role("thinker")` returns deepseek-r1-7b

---

## WBS-INF7: API Routes - Health

**Dependencies:** WBS-INF6  
**Reference:** ARCHITECTURE.md → Health Checks

### Acceptance Criteria

| ID | Requirement |
|----|-------------|
| AC-7.1 | GET /health returns 200 when service is up |
| AC-7.2 | GET /health/ready returns 200 when models loaded |
| AC-7.3 | GET /health/ready returns 503 when no models loaded |
| AC-7.4 | Health response includes loaded_models list |

### WBS Tasks

| ID | Task | AC | File(s) |
|----|------|-----|---------|
| INF7.1 | RED: Write /health tests | AC-7.1 | `tests/unit/api/routes/test_health.py` |
| INF7.2 | GREEN: Implement /health endpoint | AC-7.1 | `src/api/routes/health.py` |
| INF7.3 | RED: Write /health/ready tests | AC-7.2, AC-7.3, AC-7.4 | `tests/unit/api/routes/test_health.py` |
| INF7.4 | GREEN: Implement /health/ready endpoint | AC-7.2, AC-7.3, AC-7.4 | `src/api/routes/health.py` |
| INF7.5 | Register routes in main app | AC-7.1-7.4 | `src/main.py` |

### Exit Criteria

- [ ] `pytest tests/unit/api/routes/test_health.py` passes with 100% coverage
- [ ] `curl localhost:8085/health` returns `{"status": "ok"}`
- [ ] `curl localhost:8085/health/ready` returns model list when ready
- [ ] Kubernetes probes can use both endpoints

---

## WBS-INF8: API Routes - Models

**Dependencies:** WBS-INF6  
**Reference:** ARCHITECTURE.md → API Contract (Models endpoints)

### Acceptance Criteria

| ID | Requirement |
|----|-------------|
| AC-8.1 | GET /v1/models lists available and loaded models |
| AC-8.2 | POST /v1/models/{id}/load loads specified model |
| AC-8.3 | POST /v1/models/{id}/unload unloads specified model |
| AC-8.4 | Model response includes status, memory, context, roles |

### WBS Tasks

| ID | Task | AC | File(s) |
|----|------|-----|---------|
| INF8.1 | RED: Write GET /v1/models tests | AC-8.1, AC-8.4 | `tests/unit/api/routes/test_models.py` |
| INF8.2 | GREEN: Implement list_models endpoint | AC-8.1, AC-8.4 | `src/api/routes/models.py` |
| INF8.3 | RED: Write POST /load tests | AC-8.2 | `tests/unit/api/routes/test_models.py` |
| INF8.4 | GREEN: Implement load_model endpoint | AC-8.2 | `src/api/routes/models.py` |
| INF8.5 | RED: Write POST /unload tests | AC-8.3 | `tests/unit/api/routes/test_models.py` |
| INF8.6 | GREEN: Implement unload_model endpoint | AC-8.3 | `src/api/routes/models.py` |
| INF8.7 | Register routes in main app | AC-8.1-8.4 | `src/main.py` |

### Exit Criteria

- [ ] `pytest tests/unit/api/routes/test_models.py` passes with 100% coverage
- [ ] GET /v1/models returns JSON matching ARCHITECTURE.md schema
- [ ] Load/unload endpoints update model status correctly
- [ ] Invalid model_id returns 404 with clear error message

---

## WBS-INF9: API Routes - Completions

**Dependencies:** WBS-INF6  
**Reference:** ARCHITECTURE.md → API Contract (Chat Completions)

### Acceptance Criteria

| ID | Requirement |
|----|-------------|
| AC-9.1 | POST /v1/chat/completions accepts OpenAI-compatible request |
| AC-9.2 | Non-streaming response matches OpenAI format |
| AC-9.3 | Streaming response uses SSE format |
| AC-9.4 | Response includes usage statistics |
| AC-9.5 | Response includes orchestration metadata |

### WBS Tasks

| ID | Task | AC | File(s) |
|----|------|-----|---------|
| INF9.1 | RED: Write non-streaming completion tests | AC-9.1, AC-9.2, AC-9.4 | `tests/unit/api/routes/test_completions.py` |
| INF9.2 | GREEN: Implement chat_completions endpoint | AC-9.1, AC-9.2 | `src/api/routes/completions.py` |
| INF9.3 | Implement usage calculation | AC-9.4 | `src/api/routes/completions.py` |
| INF9.4 | RED: Write streaming completion tests | AC-9.3 | `tests/unit/api/routes/test_completions.py` |
| INF9.5 | GREEN: Implement streaming with StreamingResponse | AC-9.3 | `src/api/routes/completions.py` |
| INF9.6 | RED: Write orchestration metadata tests | AC-9.5 | `tests/unit/api/routes/test_completions.py` |
| INF9.7 | GREEN: Add orchestration field to response | AC-9.5 | `src/api/routes/completions.py` |
| INF9.8 | Implement FastAPI dependencies | AC-9.1-9.5 | `src/api/dependencies.py` |
| INF9.9 | Register routes in main app | AC-9.1-9.5 | `src/main.py` |

### Exit Criteria

- [ ] `pytest tests/unit/api/routes/test_completions.py` passes with 100% coverage
- [ ] Request JSON from ARCHITECTURE.md accepted without error
- [ ] Response JSON validates against OpenAI schema
- [ ] Streaming ends with `data: [DONE]\n\n`

---

## WBS-INF10: Context Management

**Dependencies:** WBS-INF9  
**Reference:** ARCHITECTURE.md → Context Management

### Acceptance Criteria

| ID | Requirement |
|----|-------------|
| AC-10.1 | HandoffState dataclass with default_factory (AP-1.5) |
| AC-10.2 | Token budget allocation per model |
| AC-10.3 | fit_to_budget() iterative compression (AP-2.1) |
| AC-10.4 | Trajectory injection in prompts |
| AC-10.5 | Error contamination detection |

### WBS Tasks

| ID | Task | AC | File(s) |
|----|------|-----|---------|
| INF10.1 | RED: Write HandoffState tests | AC-10.1 | `tests/unit/orchestration/test_context.py` |
| INF10.2 | GREEN: Implement HandoffState dataclass | AC-10.1 | `src/orchestration/context.py` |
| INF10.3 | RED: Write token budget tests | AC-10.2 | `tests/unit/orchestration/test_context.py` |
| INF10.4 | GREEN: Implement MODEL_CONTEXT_BUDGETS | AC-10.2 | `src/orchestration/context.py` |
| INF10.5 | RED: Write fit_to_budget tests | AC-10.3 | `tests/unit/orchestration/test_context.py` |
| INF10.6 | GREEN: Implement fit_to_budget() with helpers | AC-10.3 | `src/orchestration/context.py` |
| INF10.7 | RED: Write trajectory injection tests | AC-10.4 | `tests/unit/orchestration/test_context.py` |
| INF10.8 | GREEN: Implement inject_trajectory() | AC-10.4 | `src/orchestration/context.py` |
| INF10.9 | RED: Write error detection tests | AC-10.5 | `tests/unit/orchestration/test_context.py` |
| INF10.10 | GREEN: Implement ErrorContaminationDetector | AC-10.5 | `src/orchestration/context.py` |

### Exit Criteria

- [ ] `pytest tests/unit/orchestration/test_context.py` passes with 100% coverage
- [ ] HandoffState with empty list creates new list (not shared reference)
- [ ] fit_to_budget() uses iteration, not recursion
- [ ] Trajectory format matches ARCHITECTURE.md spec

---

## WBS-INF11: Orchestration - Single Mode

**Dependencies:** WBS-INF10  
**Reference:** ARCHITECTURE.md → Orchestration Modes (single)

### Acceptance Criteria

| ID | Requirement |
|----|-------------|
| AC-11.1 | SingleMode passes request directly to one model |
| AC-11.2 | SingleMode returns response with usage stats |
| AC-11.3 | SingleMode supports streaming |
| AC-11.4 | Orchestrator dispatches to correct mode |

### WBS Tasks

| ID | Task | AC | File(s) |
|----|------|-----|---------|
| INF11.1 | RED: Write SingleMode tests | AC-11.1, AC-11.2 | `tests/unit/orchestration/modes/test_single.py` |
| INF11.2 | GREEN: Implement SingleMode class | AC-11.1, AC-11.2 | `src/orchestration/modes/single.py` |
| INF11.3 | RED: Write SingleMode streaming tests | AC-11.3 | `tests/unit/orchestration/modes/test_single.py` |
| INF11.4 | GREEN: Implement SingleMode.stream() | AC-11.3 | `src/orchestration/modes/single.py` |
| INF11.5 | RED: Write orchestrator dispatch tests | AC-11.4 | `tests/unit/orchestration/test_orchestrator.py` |
| INF11.6 | GREEN: Implement Orchestrator class | AC-11.4 | `src/orchestration/orchestrator.py` |
| INF11.7 | Implement OrchestrationMode enum | AC-11.4 | `src/orchestration/orchestrator.py` |

### Exit Criteria

- [ ] `pytest tests/unit/orchestration/modes/test_single.py` passes with 100% coverage
- [ ] SingleMode("phi-4").execute() returns valid ChatCompletionResponse
- [ ] Orchestrator(mode="single") dispatches to SingleMode
- [ ] Streaming produces valid SSE chunks

---

## WBS-INF12: Orchestration - Critique Mode

**Dependencies:** WBS-INF11  
**Reference:** ARCHITECTURE.md → Orchestration Modes (critique)

### Acceptance Criteria

| ID | Requirement |
|----|-------------|
| AC-12.1 | CritiqueMode: Generator → Critic → Revise flow |
| AC-12.2 | CritiqueMode uses HandoffState between steps |
| AC-12.3 | CritiqueMode respects max_rounds setting |
| AC-12.4 | CritiqueMode reports models_used in metadata |

### WBS Tasks

| ID | Task | AC | File(s) |
|----|------|-----|---------|
| INF12.1 | RED: Write CritiqueMode flow tests | AC-12.1, AC-12.2 | `tests/unit/orchestration/modes/test_critique.py` |
| INF12.2 | GREEN: Implement CritiqueMode class | AC-12.1 | `src/orchestration/modes/critique.py` |
| INF12.3 | Implement generate phase | AC-12.1 | `src/orchestration/modes/critique.py` |
| INF12.4 | Implement critique phase | AC-12.1 | `src/orchestration/modes/critique.py` |
| INF12.5 | Implement revise phase | AC-12.1 | `src/orchestration/modes/critique.py` |
| INF12.6 | RED: Write max_rounds tests | AC-12.3 | `tests/unit/orchestration/modes/test_critique.py` |
| INF12.7 | GREEN: Implement round limiting | AC-12.3 | `src/orchestration/modes/critique.py` |
| INF12.8 | RED: Write metadata tests | AC-12.4 | `tests/unit/orchestration/modes/test_critique.py` |
| INF12.9 | GREEN: Populate OrchestrationMetadata | AC-12.4 | `src/orchestration/modes/critique.py` |

### Exit Criteria

- [ ] `pytest tests/unit/orchestration/modes/test_critique.py` passes with 100% coverage
- [ ] CritiqueMode executes Generator → Critic → Revise
- [ ] HandoffState carries decisions between phases
- [ ] metadata.models_used lists both models

---

## WBS-INF13: Orchestration - Pipeline Mode

**Dependencies:** WBS-INF11  
**Reference:** ARCHITECTURE.md → Orchestration Modes (pipeline), Saga Compensation

### Acceptance Criteria

| ID | Requirement |
|----|-------------|
| AC-13.1 | PipelineMode: Draft → Refine → Validate flow |
| AC-13.2 | PipelineMode compresses between steps if needed |
| AC-13.3 | PipelineMode supports saga compensation on failure |
| AC-13.4 | PipelineMode returns partial result on error |

### WBS Tasks

| ID | Task | AC | File(s) |
|----|------|-----|---------|
| INF13.1 | RED: Write PipelineMode flow tests | AC-13.1 | `tests/unit/orchestration/modes/test_pipeline.py` |
| INF13.2 | GREEN: Implement PipelineMode class | AC-13.1 | `src/orchestration/modes/pipeline.py` |
| INF13.3 | Implement draft phase | AC-13.1 | `src/orchestration/modes/pipeline.py` |
| INF13.4 | Implement refine phase | AC-13.1 | `src/orchestration/modes/pipeline.py` |
| INF13.5 | Implement validate phase | AC-13.1 | `src/orchestration/modes/pipeline.py` |
| INF13.6 | RED: Write compression handoff tests | AC-13.2 | `tests/unit/orchestration/modes/test_pipeline.py` |
| INF13.7 | GREEN: Add fit_to_budget() calls | AC-13.2 | `src/orchestration/modes/pipeline.py` |
| INF13.8 | RED: Write PipelineSaga tests | AC-13.3, AC-13.4 | `tests/unit/orchestration/test_saga.py` |
| INF13.9 | GREEN: Implement PipelineSaga class | AC-13.3, AC-13.4 | `src/orchestration/saga.py` |
| INF13.10 | Implement compensation logic | AC-13.4 | `src/orchestration/saga.py` |

### Exit Criteria

- [ ] `pytest tests/unit/orchestration/modes/test_pipeline.py` passes with 100% coverage
- [ ] PipelineMode executes Draft → Refine → Validate
- [ ] Content >16K tokens compressed before passing to phi-4
- [ ] Failed pipeline returns partial result with `finish_reason="partial"`

---

## WBS-INF14: Orchestration - Debate Mode

**Dependencies:** WBS-INF11  
**Reference:** ARCHITECTURE.md → Orchestration Modes (debate)

### Acceptance Criteria

| ID | Requirement |
|----|-------------|
| AC-14.1 | DebateMode: Parallel generation → Reconcile |
| AC-14.2 | DebateMode uses asyncio.gather for parallel |
| AC-14.3 | DebateMode compares outputs for agreement % |
| AC-14.4 | DebateMode reconciler synthesizes final answer |

### WBS Tasks

| ID | Task | AC | File(s) |
|----|------|-----|---------|
| INF14.1 | RED: Write parallel generation tests | AC-14.1, AC-14.2 | `tests/unit/orchestration/modes/test_debate.py` |
| INF14.2 | GREEN: Implement parallel generation | AC-14.1, AC-14.2 | `src/orchestration/modes/debate.py` |
| INF14.3 | RED: Write comparison tests | AC-14.3 | `tests/unit/orchestration/modes/test_debate.py` |
| INF14.4 | GREEN: Implement output comparison | AC-14.3 | `src/orchestration/modes/debate.py` |
| INF14.5 | RED: Write reconciliation tests | AC-14.4 | `tests/unit/orchestration/modes/test_debate.py` |
| INF14.6 | GREEN: Implement reconciler | AC-14.4 | `src/orchestration/modes/debate.py` |
| INF14.7 | Implement agreement percentage calculation | AC-14.3 | `src/orchestration/modes/debate.py` |

### Exit Criteria

- [ ] `pytest tests/unit/orchestration/modes/test_debate.py` passes with 100% coverage
- [ ] Both models generate in parallel (verified with timing)
- [ ] metadata.confidence reflects agreement percentage
- [ ] Reconciler creates coherent synthesis

---

## WBS-INF15: Orchestration - Ensemble Mode

**Dependencies:** WBS-INF11  
**Reference:** ARCHITECTURE.md → Orchestration Modes (ensemble)

### Acceptance Criteria

| ID | Requirement |
|----|-------------|
| AC-15.1 | EnsembleMode: All models generate in parallel |
| AC-15.2 | EnsembleMode calculates consensus score |
| AC-15.3 | EnsembleMode synthesizes from agreed points |
| AC-15.4 | EnsembleMode flags disagreements |

### WBS Tasks

| ID | Task | AC | File(s) |
|----|------|-----|---------|
| INF15.1 | RED: Write parallel ensemble tests | AC-15.1 | `tests/unit/orchestration/modes/test_ensemble.py` |
| INF15.2 | GREEN: Implement parallel generation | AC-15.1 | `src/orchestration/modes/ensemble.py` |
| INF15.3 | RED: Write consensus tests | AC-15.2 | `tests/unit/orchestration/modes/test_ensemble.py` |
| INF15.4 | GREEN: Implement consensus calculation | AC-15.2 | `src/orchestration/modes/ensemble.py` |
| INF15.5 | RED: Write synthesis tests | AC-15.3 | `tests/unit/orchestration/modes/test_ensemble.py` |
| INF15.6 | GREEN: Implement point synthesis | AC-15.3 | `src/orchestration/modes/ensemble.py` |
| INF15.7 | RED: Write disagreement tests | AC-15.4 | `tests/unit/orchestration/modes/test_ensemble.py` |
| INF15.8 | GREEN: Implement disagreement flagging | AC-15.4 | `src/orchestration/modes/ensemble.py` |

### Exit Criteria

- [ ] `pytest tests/unit/orchestration/modes/test_ensemble.py` passes with 100% coverage
- [ ] 3+ models generate in parallel
- [ ] metadata.confidence >= 0.7 for valid consensus
- [ ] Disagreements listed in response metadata

---

## WBS-INF16: Caching Infrastructure

**Dependencies:** WBS-INF10  
**Reference:** ARCHITECTURE.md → Caching Strategy

### Acceptance Criteria

| ID | Requirement |
|----|-------------|
| AC-16.1 | InferenceCache ABC with required methods |
| AC-16.2 | PromptCache caches tokenized prompts |
| AC-16.3 | HandoffCache with asyncio.Lock (AP-10.1) |
| AC-16.4 | CompressionCache for compressed content |
| AC-16.5 | CacheInvalidator invalidates on model change |

### WBS Tasks

| ID | Task | AC | File(s) |
|----|------|-----|---------|
| INF16.1 | RED: Write InferenceCache ABC tests | AC-16.1 | `tests/unit/services/test_cache.py` |
| INF16.2 | GREEN: Implement InferenceCache ABC | AC-16.1 | `src/services/cache.py` |
| INF16.3 | RED: Write PromptCache tests | AC-16.2 | `tests/unit/services/test_cache.py` |
| INF16.4 | GREEN: Implement PromptCache | AC-16.2 | `src/services/cache.py` |
| INF16.5 | RED: Write HandoffCache tests | AC-16.3 | `tests/unit/services/test_cache.py` |
| INF16.6 | GREEN: Implement HandoffCache with locks | AC-16.3 | `src/services/cache.py` |
| INF16.7 | RED: Write CompressionCache tests | AC-16.4 | `tests/unit/services/test_cache.py` |
| INF16.8 | GREEN: Implement CompressionCache | AC-16.4 | `src/services/cache.py` |
| INF16.9 | RED: Write CacheInvalidator tests | AC-16.5 | `tests/unit/services/test_cache.py` |
| INF16.10 | GREEN: Implement CacheInvalidator | AC-16.5 | `src/services/cache.py` |

### Exit Criteria

- [ ] `pytest tests/unit/services/test_cache.py` passes with 100% coverage
- [ ] HandoffCache concurrent writes don't lose data (race condition test)
- [ ] CacheInvalidator clears cache on model hash change
- [ ] All caches implement InferenceCache interface

---

## WBS-INF17: Queue Manager

**Dependencies:** WBS-INF9  
**Reference:** ARCHITECTURE.md → Concurrency & Scaling

### Acceptance Criteria

| ID | Requirement |
|----|-------------|
| AC-17.1 | QueueManager uses asyncio.Queue |
| AC-17.2 | QueueManager enforces max_concurrent_requests |
| AC-17.3 | QueueManager supports FIFO and priority strategies |
| AC-17.4 | QueueManager rejects when full if configured |

### WBS Tasks

| ID | Task | AC | File(s) |
|----|------|-----|---------|
| INF17.1 | RED: Write queue basic tests | AC-17.1, AC-17.2 | `tests/unit/services/test_queue_manager.py` |
| INF17.2 | GREEN: Implement QueueManager class | AC-17.1, AC-17.2 | `src/services/queue_manager.py` |
| INF17.3 | RED: Write FIFO strategy tests | AC-17.3 | `tests/unit/services/test_queue_manager.py` |
| INF17.4 | GREEN: Implement FIFO strategy | AC-17.3 | `src/services/queue_manager.py` |
| INF17.5 | RED: Write priority strategy tests | AC-17.3 | `tests/unit/services/test_queue_manager.py` |
| INF17.6 | GREEN: Implement priority strategy | AC-17.3 | `src/services/queue_manager.py` |
| INF17.7 | RED: Write reject-when-full tests | AC-17.4 | `tests/unit/services/test_queue_manager.py` |
| INF17.8 | GREEN: Implement rejection logic | AC-17.4 | `src/services/queue_manager.py` |

### Exit Criteria

- [ ] `pytest tests/unit/services/test_queue_manager.py` passes with 100% coverage
- [ ] 11th request blocked when max=10
- [ ] Priority=3 processed before priority=1
- [ ] QueueFullError raised when reject_when_full=True

---

## WBS-INF18: Error Handling

**Dependencies:** WBS-INF9  
**Reference:** ARCHITECTURE.md → Error Handling

### Acceptance Criteria

| ID | Requirement |
|----|-------------|
| AC-18.1 | Exception hierarchy with Retriable/NonRetriable |
| AC-18.2 | All custom exceptions end in "Error" (AP-7) |
| AC-18.3 | Error responses match llm-gateway schema |
| AC-18.4 | FastAPI exception handlers registered |

### WBS Tasks

| ID | Task | AC | File(s) |
|----|------|-----|---------|
| INF18.1 | RED: Write exception hierarchy tests | AC-18.1, AC-18.2 | `tests/unit/core/test_exceptions.py` |
| INF18.2 | GREEN: Implement exception classes | AC-18.1, AC-18.2 | `src/core/exceptions.py` |
| INF18.3 | Implement InferenceServiceError base | AC-18.1 | `src/core/exceptions.py` |
| INF18.4 | Implement RetriableError subclass | AC-18.1 | `src/core/exceptions.py` |
| INF18.5 | Implement NonRetriableError subclass | AC-18.1 | `src/core/exceptions.py` |
| INF18.6 | Implement all specific errors | AC-18.2 | `src/core/exceptions.py` |
| INF18.7 | RED: Write error response tests | AC-18.3 | `tests/unit/api/test_error_handlers.py` |
| INF18.8 | GREEN: Implement error response format | AC-18.3 | `src/api/error_handlers.py` |
| INF18.9 | GREEN: Register exception handlers | AC-18.4 | `src/main.py` |

### Exit Criteria

- [ ] `pytest tests/unit/core/test_exceptions.py` passes with 100% coverage
- [ ] All exception names match regex `.*Error$`
- [ ] Error response JSON matches ARCHITECTURE.md schema
- [ ] FastAPI returns proper status codes (400, 404, 500, 503)

---

## WBS-INF19: Docker & CI/CD

**Dependencies:** WBS-INF9  
**Reference:** ARCHITECTURE.md → Docker Health Check, Folder Structure

### Acceptance Criteria

| ID | Requirement |
|----|-------------|
| AC-19.1 | Dockerfile builds successfully |
| AC-19.2 | Docker health check uses /health endpoint |
| AC-19.3 | docker-compose.yml starts service |
| AC-19.4 | CI workflow runs tests on PR |
| AC-19.5 | CI passes with pytest, mypy, ruff |

### WBS Tasks

| ID | Task | AC | File(s) |
|----|------|-----|---------|
| INF19.1 | Create Dockerfile | AC-19.1, AC-19.2 | `docker/Dockerfile` |
| INF19.2 | Create Dockerfile.cuda | AC-19.1 | `docker/Dockerfile.cuda` |
| INF19.3 | Create docker-compose.yml | AC-19.3 | `docker/docker-compose.yml` |
| INF19.4 | Create CI workflow | AC-19.4, AC-19.5 | `.github/workflows/ci.yml` |
| INF19.5 | Create release workflow | AC-19.4 | `.github/workflows/release.yml` |
| INF19.6 | Test docker build | AC-19.1 | Manual |
| INF19.7 | Test docker-compose up | AC-19.3 | Manual |

### Exit Criteria

- [ ] `docker build -f docker/Dockerfile .` succeeds
- [ ] `docker-compose -f docker/docker-compose.yml up` starts service
- [ ] Health check passes in container
- [ ] GitHub Actions workflow shows green on test PR

---

## WBS-INF20: Anti-Pattern Compliance

**Dependencies:** All prior WBS blocks  
**Reference:** CODING_PATTERNS_ANALYSIS.md

### Acceptance Criteria

| ID | Rule | Requirement |
|----|------|-------------|
| AC-20.1 | S1192 | No duplicated string literals (extract to constants) |
| AC-20.2 | S3776 | All functions cognitive complexity < 15 |
| AC-20.3 | AP-7 | Exception classes end in "Error" |
| AC-20.4 | AP-1.5 | No mutable default arguments |
| AC-20.5 | AP-10.1 | Async caches use per-resource locks |
| AC-20.6 | Type Annotations | mypy --strict passes with 0 errors |

### WBS Tasks

| ID | Task | AC | File(s) | Tool |
|----|------|-----|---------|------|
| INF20.1 | Audit for duplicated string literals | AC-20.1 | All src/*.py | SonarLint |
| INF20.2 | Extract duplicates to constants | AC-20.1 | `src/core/constants.py` | Manual |
| INF20.3 | Audit function complexity | AC-20.2 | All src/*.py | SonarLint |
| INF20.4 | Refactor functions with CC >= 15 | AC-20.2 | As needed | Manual |
| INF20.5 | Verify exception names | AC-20.3 | `src/core/exceptions.py` | grep |
| INF20.6 | Verify no mutable defaults | AC-20.4 | All dataclasses | grep/AST |
| INF20.7 | Verify cache locks | AC-20.5 | `src/services/cache.py` | Code review |
| INF20.8 | Run mypy --strict | AC-20.6 | `src/` | mypy |
| INF20.9 | Fix all mypy errors | AC-20.6 | As needed | Manual |
| INF20.10 | Run ruff check | AC-20.6 | `src/` | ruff |

### Exit Criteria

- [ ] SonarLint: 0 S1192 issues
- [ ] SonarLint: 0 S3776 issues (max CC < 15)
- [ ] `grep -r "class.*Exception" src/` returns 0 results (all end in Error)
- [ ] `grep -r "= \[\]" src/` returns 0 results in dataclasses
- [ ] `mypy --strict src/` reports 0 errors
- [ ] `ruff check src/` reports 0 errors

---

## WBS-INF21: Integration Testing

**Dependencies:** All prior WBS blocks  
**Reference:** ARCHITECTURE.md (all sections)

### Acceptance Criteria

| ID | Requirement |
|----|-------------|
| AC-21.1 | D4 preset loads: deepseek-r1-7b + qwen2.5-7b (9.2GB total) |
| AC-21.2 | Critique mode test: qwen2.5-7b generates, deepseek-r1-7b critiques |
| AC-21.3 | Streaming test: SSE chunks received correctly ending with `[DONE]` |
| AC-21.4 | Model connectivity: "hello" ping test for each available model |
| AC-21.5 | Load test: 5 concurrent requests handled without queue overflow |

### WBS Tasks

| ID | Task | AC | File(s) |
|----|------|-----|---------|
| INF21.1 | Write D4 preset load test | AC-21.1 | `tests/integration/test_d4_preset.py` |
| INF21.2 | Write critique mode e2e test | AC-21.2 | `tests/integration/test_d4_critique.py` |
| INF21.3 | Write streaming test | AC-21.3 | `tests/integration/test_streaming.py` |
| INF21.4 | Write model connectivity "hello" tests | AC-21.4 | `tests/integration/test_model_connectivity.py` |
| INF21.5 | Write concurrent request load test | AC-21.5 | `tests/integration/test_load.py` |
| INF21.6 | Create integration test fixtures | AC-21.1-5 | `tests/integration/conftest.py` |
| INF21.7 | Document integration test setup | AC-21.1-5 | `tests/integration/README.md` |

### D4 Configuration Reference

```yaml
D4:
  name: "Thinking + Code"
  models: [deepseek-r1-7b, qwen2.5-7b]
  total_size_gb: 9.2
  orchestration_mode: critique
  roles:
    qwen2.5-7b: generator     # Creates initial code/response
    deepseek-r1-7b: critic    # Chain-of-thought critique
  hardware: "Mac 16GB (comfortable)"

# D4v2 - Upgraded with Qwen3 (added 2025-12-31)
D4v2:
  name: "Thinking + Code (Qwen3)"
  models: [deepseek-r1-7b, qwen3-8b]
  total_size_gb: 9.4
  orchestration_mode: critique
  roles:
    qwen3-8b: generator       # Qwen3-based code generation
    deepseek-r1-7b: critic    # Chain-of-thought critique
  hardware: "Mac 16GB (comfortable)"
```

### Qwen3 Model Options (Added 2025-12-31)

| Model | Size | Type | Use Case |
|-------|------|------|----------|
| qwen3-8b | 4.7GB | Dense | D4v2 preset - pair with deepseek for critique mode |
| qwen3-coder-30b-a3b | 14GB | MoE | S10 standalone - 3.3B active params per token |

### Model Connectivity Tests

**Design Principle:** All tests use HTTP API calls via configurable `INFERENCE_BASE_URL` - portable to any deployment (local, server, AWS).

```bash
# Local testing (default)
export INFERENCE_BASE_URL=http://localhost:8085

# Server deployment
export INFERENCE_BASE_URL=http://inference.internal:8085

# AWS deployment
export INFERENCE_BASE_URL=https://inference.prod.example.com
```

| Model | Size | Purpose | API Test |
|-------|------|---------|----------|
| deepseek-r1-7b | 4.7GB | CoT Thinker | `POST /v1/chat/completions` with model param |
| qwen2.5-7b | 4.5GB | Coder | `POST /v1/chat/completions` with model param |
| phi-4 | 8.4GB | General | `POST /v1/chat/completions` with model param |
| llama-3.2-3b | 2.0GB | Fast | `POST /v1/chat/completions` with model param |
| phi-3-medium-128k | 8.6GB | Long Context | `POST /v1/chat/completions` with model param |
| granite-8b-code-128k | 4.5GB | Code Analysis | `POST /v1/chat/completions` with model param |

**Test Flow (per model):**
1. `GET /v1/models` → verify model exists in available list
2. `POST /v1/models/{model}/load` → load model if not already loaded
3. `POST /v1/chat/completions` → send "Hello, respond briefly" prompt
4. Assert: response contains non-empty `choices[0].message.content`
5. Assert: response contains valid `usage.total_tokens` > 0

### Portability Requirements

| Requirement | Implementation |
|-------------|----------------|
| Configurable URL | `INFERENCE_BASE_URL` env var (default: `http://localhost:8085`) |
| No hardcoded hosts | All tests use `{base_url}/v1/...` pattern |
| Model discovery | Tests query `/v1/models` to find available models |
| Skip unavailable | `pytest.mark.skipif` when model not in deployment |
| Timeout handling | Configurable `INFERENCE_TEST_TIMEOUT` (default: 120s) |
| Auth support | Optional `INFERENCE_API_KEY` header for secured deployments |

### Exit Criteria

- [ ] `INFERENCE_DEFAULT_PRESET=D4 pytest tests/integration/test_d4_preset.py` passes
- [ ] D4 loads deepseek-r1-7b + qwen2.5-7b within memory budget
- [ ] Critique mode response includes `orchestration.models_used` with both models
- [ ] SSE chunks end with `data: [DONE]\n\n`
- [ ] Each model connectivity test returns a valid response via HTTP API
- [ ] 5 concurrent requests complete without `QueueFullError`
- [ ] **Portability:** `INFERENCE_BASE_URL=http://remote:8085 pytest tests/integration/` works
- [ ] **Smoke Test:** Can run `pytest tests/integration/test_model_connectivity.py -v` as deployment verification

---

## Implementation Order

```
WBS-INF1 (Scaffolding)
    │
    ▼
WBS-INF2 (Core Infrastructure)
    │
    ▼
WBS-INF3 (Pydantic Models)
    │
    ▼
WBS-INF4 (Provider Abstraction)
    │
    ▼
WBS-INF5 (LlamaCpp Provider)
    │
    ▼
WBS-INF6 (Model Manager)
    │
    ├──────────────┬──────────────┐
    ▼              ▼              ▼
WBS-INF7       WBS-INF8       WBS-INF9
(Health)       (Models)       (Completions)
    │              │              │
    └──────────────┴──────────────┘
                   │
    ┌──────────────┼──────────────┐
    ▼              ▼              ▼
WBS-INF10      WBS-INF17      WBS-INF18
(Context)      (Queue)        (Errors)
    │
    ▼
WBS-INF11 (Single Mode)
    │
    ├──────────┬──────────┬──────────┐
    ▼          ▼          ▼          ▼
WBS-INF12  WBS-INF13  WBS-INF14  WBS-INF15
(Critique) (Pipeline) (Debate)   (Ensemble)
    │          │          │          │
    └──────────┴──────────┴──────────┘
                   │
    ┌──────────────┤
    ▼              ▼
WBS-INF16      WBS-INF19
(Caching)      (Docker/CI)
    │              │
    └──────────────┘
           │
           ▼
    WBS-INF20 (Anti-Pattern)
           │
           ▼
    WBS-INF21 (Integration)
```

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | 2025-12-27 | Initial WBS based on ARCHITECTURE.md v1.3.1 |
