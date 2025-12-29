# Integration Tests

This directory contains integration tests for the inference-service.

## Overview

Integration tests verify end-to-end behavior of the inference-service with actual HTTP requests and model inference.

## Test Files

| File | Purpose | AC |
|------|---------|-----|
| `test_e2e.py` | End-to-end chat completion tests | AC-21.1 |
| `test_streaming.py` | SSE streaming response tests | AC-21.2 |
| `test_orchestration.py` | Multi-model orchestration tests | AC-21.3 |
| `test_gateway.py` | llm-gateway integration tests | AC-21.4 |
| `test_load.py` | Concurrent request load tests | AC-21.5 |
| `conftest.py` | Shared fixtures and configuration | - |

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

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `INFERENCE_SERVICE_URL` | `http://localhost:8085` | URL of inference-service |
| `LLM_GATEWAY_URL` | `http://localhost:8080` | URL of llm-gateway |
| `INTEGRATION_TEST_TIMEOUT` | `120.0` | Default request timeout (seconds) |
| `STREAM_TEST_TIMEOUT` | `180.0` | Streaming request timeout |

### Required Models

For full test coverage, at least one model should be loaded:

- `llama-3.2-3b` (recommended for fast tests)
- `phi-4` (for orchestration tests)
- `deepseek-r1-7b` (for multi-model tests)

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

- [ ] `pytest tests/integration/ -m "not slow"` passes
- [ ] SSE chunks end with `data: [DONE]\n\n`
- [ ] Critique mode uses 2+ models
- [ ] `inference:phi-4` routes through gateway
- [ ] 10 concurrent requests complete within timeout

## Troubleshooting

### Service Not Available

```
SKIPPED: Inference service not available at http://localhost:8085
```

**Solution:** Start the inference-service before running tests.

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

**Solution:** Increase timeout via `INTEGRATION_TEST_TIMEOUT` environment variable.
