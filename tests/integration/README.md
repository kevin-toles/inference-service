# Integration Tests

This directory contains integration tests for the inference-service.

## Overview

Integration tests verify end-to-end behavior of the inference-service with actual HTTP requests and model inference. **All tests are portable** - they use HTTP API calls and can run against any deployment (local, server, AWS).

## Test Files

| File | Purpose | AC |
|------|---------|-----|
| `test_d4_preset.py` | D4 preset load tests (deepseek + qwen) | AC-21.1 |
| `test_d4_critique.py` | Critique mode e2e tests | AC-21.2 |
| `test_streaming.py` | SSE streaming response tests | AC-21.3 |
| `test_model_connectivity.py` | Model "hello" ping tests | AC-21.4 |
| `test_load.py` | Concurrent request load tests | AC-21.5 |
| `test_e2e.py` | End-to-end chat completion tests | - |
| `test_orchestration.py` | Multi-model orchestration tests | - |
| `test_gateway.py` | llm-gateway integration tests | - |
| `conftest.py` | Shared fixtures and configuration | - |

## Quick Start

```bash
# Run all integration tests (local)
pytest tests/integration/ -v

# Run just connectivity smoke tests
pytest tests/integration/test_model_connectivity.py -v

# Run D4 preset tests
INFERENCE_DEFAULT_PRESET=D4 pytest tests/integration/test_d4_preset.py -v
```

## Portable Deployment Testing

### Design Principle

All tests use HTTP API calls via configurable `INFERENCE_BASE_URL` - no hardcoded hosts, no direct Python imports. This makes tests portable across:

- Local Mac development
- On-premise servers
- AWS/Cloud deployments
- Docker containers

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `INFERENCE_BASE_URL` | `http://localhost:8085` | **Primary URL** - use this for portability |
| `INFERENCE_SERVICE_URL` | (same as BASE_URL) | Backward compatibility alias |
| `INFERENCE_API_KEY` | (empty) | Optional bearer token for secured deployments |
| `INFERENCE_TEST_TIMEOUT` | `120.0` | Default request timeout (seconds) |
| `STREAM_TEST_TIMEOUT` | `180.0` | Streaming request timeout |
| `MODEL_LOAD_TIMEOUT` | `300.0` | Timeout for model loading operations |
| `LLM_GATEWAY_URL` | `http://localhost:8080` | URL of llm-gateway (optional) |

### Usage Examples

```bash
# Local Mac development (default)
pytest tests/integration/test_model_connectivity.py -v

# On-premise server
INFERENCE_BASE_URL=http://10.0.0.50:8085 \
pytest tests/integration/test_model_connectivity.py -v

# AWS deployment with authentication
INFERENCE_BASE_URL=https://inference.prod.example.com \
INFERENCE_API_KEY=sk-your-api-key \
pytest tests/integration/test_model_connectivity.py -v

# Docker Compose internal network
INFERENCE_BASE_URL=http://inference-service:8085 \
pytest tests/integration/test_model_connectivity.py -v
```

## Model Connectivity Tests

The `test_model_connectivity.py` file provides **deployment smoke tests** for all available models:

| Model | Size | Purpose | Test |
|-------|------|---------|------|
| deepseek-r1-7b | 4.7GB | CoT Thinker | "Hello, respond briefly" |
| qwen2.5-7b | 4.5GB | Coder | "Hello, respond briefly" |
| phi-4 | 8.4GB | General | "Hello, respond briefly" |
| llama-3.2-3b | 2.0GB | Fast | "Hello, respond briefly" |
| phi-3-medium-128k | 8.6GB | Long Context | "Hello, respond briefly" |
| granite-8b-code-128k | 4.5GB | Code Analysis | "Hello, respond briefly" |

### Test Flow (per model)

1. `GET /v1/models` → verify model exists in available list
2. `POST /v1/models/{model}/load` → load model if not already loaded
3. `POST /v1/chat/completions` → send "Hello, respond briefly" prompt
4. Assert: response contains non-empty `choices[0].message.content`
5. Assert: response contains valid `usage.total_tokens` > 0

### Deployment Verification

Use this command as a deployment smoke test:

```bash
# Quick deployment verification
INFERENCE_BASE_URL=http://your-deployment:8085 \
pytest tests/integration/test_model_connectivity.py::TestDeploymentSummary -v -s

# Full model connectivity check
INFERENCE_BASE_URL=http://your-deployment:8085 \
pytest tests/integration/test_model_connectivity.py -v
```

## D4 Preset Tests

D4 is a dual-model configuration for critique mode:

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
```

### Running D4 Tests

```bash
# Load D4 preset and run tests
INFERENCE_DEFAULT_PRESET=D4 pytest tests/integration/test_d4_preset.py -v

# Run critique mode tests
INFERENCE_DEFAULT_PRESET=D4 pytest tests/integration/test_d4_critique.py -v
```

## Requirements

### Running inference-service

The inference-service must be running and accessible:

```bash
# Start the service
cd /path/to/inference-service
python -m uvicorn src.main:app --host 0.0.0.0 --port 8085

# Or with Docker
docker-compose -f docker/docker-compose.yml up -d
```

### Required Models

For full test coverage, at least one model should be loaded:

- `llama-3.2-3b` (recommended for fast tests)
- `phi-4` (for orchestration tests)
- `deepseek-r1-7b` + `qwen2.5-7b` (for D4/critique tests)

## Running Tests

### Run All Integration Tests

```bash
# Requires running inference-service
pytest tests/integration/ -v

# With markers
pytest tests/integration/ -v -m integration
```

### Skip Slow Tests

```bash
# Skip long-running tests (load tests, multi-model)
pytest tests/integration/ -v -m "not slow"
```

### Run Specific Test Categories

```bash
# E2E tests only
pytest tests/integration/test_e2e.py -v

# Streaming tests
pytest tests/integration/test_streaming.py -v

# Orchestration tests
pytest tests/integration/test_orchestration.py -v

# Gateway tests (requires llm-gateway running)
pytest tests/integration/test_gateway.py -v

# Load tests
pytest tests/integration/test_load.py -v
```

### Run Tests Requiring Models

```bash
# Only tests that need loaded models
pytest tests/integration/ -v -m requires_model
```

### Run Gateway Tests

```bash
# Only tests requiring llm-gateway
pytest tests/integration/ -v -m requires_gateway
```

## Test Markers

| Marker | Description |
|--------|-------------|
| `@pytest.mark.integration` | All integration tests |
| `@pytest.mark.slow` | Long-running tests |
| `@pytest.mark.requires_model` | Requires loaded model |
| `@pytest.mark.requires_gateway` | Requires llm-gateway |

## Fixtures

### HTTP Clients

- `client` - Per-test async HTTP client for inference-service
- `async_client` - Session-scoped async client
- `gateway_client` - Async client for llm-gateway

### Skip Conditions

- `skip_if_service_unavailable` - Skip if service not reachable
- `skip_if_gateway_unavailable` - Skip if gateway not reachable
- `skip_if_no_models` - Skip if no models loaded

### Request Builders

- `chat_request_factory` - Factory for building chat requests
  - `.simple()` - Basic chat request
  - `.with_system()` - Request with system message
  - `.streaming()` - Streaming request
  - `.orchestrated()` - Request with orchestration mode
  - `.code_task()` - Code generation request

### Validators

- `response_validator` - Validate response structure
  - `.validate_completion()` - Validate non-streaming response
  - `.validate_usage()` - Validate usage statistics
  - `.validate_orchestration()` - Validate orchestration metadata
  - `.validate_streaming_chunk()` - Validate streaming chunk

### Parsers

- `sse_parser` - Parse SSE streams
  - `.parse_line()` - Parse single SSE line
  - `.parse_stream()` - Parse entire stream
  - `.extract_content()` - Extract content from chunks

## Example Test

```python
@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.requires_model
async def test_simple_completion(
    client: httpx.AsyncClient,
    skip_if_no_models: None,
    chat_request_factory: type,
    response_validator: type,
) -> None:
    """Test simple chat completion."""
    request = chat_request_factory.simple(
        message="What is 2 + 2?",
        max_tokens=50,
    )
    
    response = await client.post("/v1/chat/completions", json=request)
    
    assert response.status_code == 200
    data = response.json()
    response_validator.validate_completion(data)
```

## Exit Criteria

- [x] `INFERENCE_DEFAULT_PRESET=D4 pytest tests/integration/test_d4_preset.py` passes
- [x] D4 loads deepseek-r1-7b + qwen2.5-7b within memory budget
- [x] Critique mode response includes `orchestration.models_used` with both models
- [x] SSE chunks end with `data: [DONE]\n\n`
- [x] Each model connectivity test returns a valid response via HTTP API
- [x] 5 concurrent requests complete without `QueueFullError`
- [x] **Portability:** `INFERENCE_BASE_URL=http://remote:8085 pytest tests/integration/` works
- [x] **Smoke Test:** Can run `pytest tests/integration/test_model_connectivity.py -v` as deployment verification

## Troubleshooting

### Service Not Available

```
SKIPPED: Inference service not available at http://localhost:8085
```

**Solution:** Start the inference-service before running tests.

### Model Not Available

```
pytest.skip: Model deepseek-r1-7b not available in this deployment
```

**Solution:** The model isn't configured in this deployment. This is expected - tests skip gracefully.

### No Models Loaded

```
SKIPPED: No models loaded in inference service
```

**Solution:** Load at least one model via `/v1/models/{id}/load` or configure auto-loading.

### Gateway Tests Skipped

```
SKIPPED: LLM Gateway not available at http://localhost:8080
```

**Solution:** Start llm-gateway if testing gateway integration.

### Timeout Errors

```
httpx.TimeoutException: timed out
```

**Solution:** Increase timeout via `INFERENCE_TEST_TIMEOUT` environment variable:

```bash
INFERENCE_TEST_TIMEOUT=300 pytest tests/integration/ -v
```

### Authentication Errors

```
401 Unauthorized
```

**Solution:** Set API key for secured deployments:

```bash
INFERENCE_API_KEY=your-key pytest tests/integration/ -v
```

## CI/CD Integration

### GitHub Actions Example

```yaml
jobs:
  integration-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Start inference-service
        run: docker-compose -f docker/docker-compose.yml up -d
        
      - name: Wait for service
        run: |
          for i in {1..30}; do
            curl -s http://localhost:8085/health && break
            sleep 2
          done
          
      - name: Run integration tests
        env:
          INFERENCE_BASE_URL: http://localhost:8085
        run: pytest tests/integration/ -v --tb=short
```

### AWS Deployment Testing

```yaml
# In your AWS deployment pipeline
- name: Smoke Test Deployment
  env:
    INFERENCE_BASE_URL: ${{ secrets.INFERENCE_PROD_URL }}
    INFERENCE_API_KEY: ${{ secrets.INFERENCE_API_KEY }}
  run: |
    pytest tests/integration/test_model_connectivity.py -v
```
