"""Integration test fixtures and configuration.

This module provides shared fixtures for integration tests including:
- HTTP client for API requests
- Service lifecycle management
- Model loading helpers
- Test data generators

Usage:
    pytest tests/integration/ -m "not slow"  # Skip slow tests
    pytest tests/integration/ --integration  # All integration tests
    pytest tests/integration/ -k "e2e"       # Run only e2e tests

Requirements:
    - inference-service running on INFERENCE_SERVICE_URL (default: http://localhost:8085)
    - For gateway tests: llm-gateway running on LLM_GATEWAY_URL (default: http://localhost:8080)
    - For model tests: At least one GGUF model available
"""

from __future__ import annotations

import asyncio
import contextlib
import os
from collections.abc import AsyncGenerator, Generator
from typing import Any

import httpx
import pytest


# =============================================================================
# Configuration - PORTABLE across deployments (local, server, AWS)
# =============================================================================

# Primary URL configuration - use INFERENCE_BASE_URL for portability
INFERENCE_BASE_URL = os.getenv("INFERENCE_BASE_URL", "http://localhost:8085")
# Backward compatibility alias
INFERENCE_SERVICE_URL = os.getenv("INFERENCE_SERVICE_URL", INFERENCE_BASE_URL)
LLM_GATEWAY_URL = os.getenv("LLM_GATEWAY_URL", "http://localhost:8080")
DEFAULT_TIMEOUT = float(os.getenv("INFERENCE_TEST_TIMEOUT", "120.0"))
STREAM_TIMEOUT = float(os.getenv("STREAM_TEST_TIMEOUT", "180.0"))

# Optional API key for secured deployments (AWS, production)
INFERENCE_API_KEY = os.getenv("INFERENCE_API_KEY", "")

# Model timeout for large models (e.g., phi-4, phi-3-medium-128k)
MODEL_LOAD_TIMEOUT = float(os.getenv("MODEL_LOAD_TIMEOUT", "300.0"))

# =============================================================================
# Constants (S1192: Avoid duplicated string literals)
# =============================================================================

TEST_MODEL_LLAMA = "llama-3.2-3b"
TEST_MODEL_PHI4 = "phi-4"
TEST_MODEL_QWEN = "qwen2.5-7b"
TEST_CONTENT_TYPE_JSON = "application/json"
AUTH_HEADER_BEARER = "Bearer"
ROLE_USER = "user"
ROLE_SYSTEM = "system"
SSE_DONE_MARKER = "[DONE]"
SSE_DATA_PREFIX = "data: "


# =============================================================================
# Markers
# =============================================================================


def pytest_configure(config: pytest.Config) -> None:
    """Register custom pytest markers."""
    config.addinivalue_line(
        "markers", "integration: mark test as integration test (requires running services)"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow (skip with -m 'not slow')"
    )
    config.addinivalue_line(
        "markers", "requires_model: mark test as requiring a loaded model"
    )
    config.addinivalue_line(
        "markers", "requires_gateway: mark test as requiring llm-gateway"
    )


# =============================================================================
# Service Health Fixtures
# =============================================================================


@pytest.fixture(scope="session")
def event_loop() -> Generator[asyncio.AbstractEventLoop, None, None]:
    """Create an event loop for the test session."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
def inference_service_url() -> str:
    """Return the inference service URL (portable via INFERENCE_BASE_URL)."""
    return INFERENCE_BASE_URL


@pytest.fixture(scope="session")
def inference_base_url() -> str:
    """Return the inference base URL for portable tests."""
    return INFERENCE_BASE_URL


@pytest.fixture(scope="session")
def api_headers() -> dict[str, str]:
    """Return API headers including optional auth for secured deployments."""
    headers: dict[str, str] = {"Content-Type": TEST_CONTENT_TYPE_JSON}
    if INFERENCE_API_KEY:
        headers["Authorization"] = f"{AUTH_HEADER_BEARER} {INFERENCE_API_KEY}"
    return headers


@pytest.fixture(scope="session")
def gateway_url() -> str:
    """Return the llm-gateway URL."""
    return LLM_GATEWAY_URL


@pytest.fixture(scope="session")
async def async_client() -> AsyncGenerator[httpx.AsyncClient, None]:
    """Create an async HTTP client for the test session (portable)."""
    headers: dict[str, str] = {}
    if INFERENCE_API_KEY:
        headers["Authorization"] = f"{AUTH_HEADER_BEARER} {INFERENCE_API_KEY}"

    async with httpx.AsyncClient(
        base_url=INFERENCE_BASE_URL,
        timeout=httpx.Timeout(DEFAULT_TIMEOUT),
        headers=headers,
    ) as client:
        yield client


@pytest.fixture(scope="session")
async def gateway_client() -> AsyncGenerator[httpx.AsyncClient, None]:
    """Create an async HTTP client for llm-gateway."""
    async with httpx.AsyncClient(
        base_url=LLM_GATEWAY_URL,
        timeout=httpx.Timeout(DEFAULT_TIMEOUT),
    ) as client:
        yield client


@pytest.fixture(scope="function")
async def client() -> AsyncGenerator[httpx.AsyncClient, None]:
    """Create a per-test async HTTP client (portable)."""
    headers: dict[str, str] = {}
    if INFERENCE_API_KEY:
        headers["Authorization"] = f"{AUTH_HEADER_BEARER} {INFERENCE_API_KEY}"

    async with httpx.AsyncClient(
        base_url=INFERENCE_BASE_URL,
        timeout=httpx.Timeout(DEFAULT_TIMEOUT),
        headers=headers,
    ) as client:
        yield client


# =============================================================================
# Service Availability Fixtures
# =============================================================================


@pytest.fixture(scope="session")
async def service_available(async_client: httpx.AsyncClient) -> bool:
    """Check if the inference service is available."""
    try:
        response = await async_client.get("/health")
        return response.status_code == 200
    except httpx.ConnectError:
        return False


@pytest.fixture(scope="session")
async def gateway_available(gateway_client: httpx.AsyncClient) -> bool:
    """Check if the llm-gateway is available."""
    try:
        response = await gateway_client.get("/health")
        return response.status_code == 200
    except httpx.ConnectError:
        return False


@pytest.fixture(scope="session")
async def service_ready(async_client: httpx.AsyncClient) -> bool:
    """Check if the inference service has models loaded."""
    try:
        response = await async_client.get("/health/ready")
        return response.status_code == 200
    except httpx.ConnectError:
        return False


# =============================================================================
# Skip Conditions
# =============================================================================


@pytest.fixture
def skip_if_service_unavailable(service_available: bool) -> None:
    """Skip test if inference service is not available."""
    if not service_available:
        pytest.skip(f"Inference service not available at {INFERENCE_SERVICE_URL}")


@pytest.fixture
def skip_if_gateway_unavailable(gateway_available: bool) -> None:
    """Skip test if llm-gateway is not available."""
    if not gateway_available:
        pytest.skip(f"LLM Gateway not available at {LLM_GATEWAY_URL}")


@pytest.fixture
def skip_if_no_models(service_ready: bool) -> None:
    """Skip test if no models are loaded."""
    if not service_ready:
        pytest.skip("No models loaded in inference service")


# =============================================================================
# Request Builders
# =============================================================================


@pytest.fixture
def chat_request_factory() -> Any:
    """Factory for creating chat completion requests."""

    class ChatRequestFactory:
        """Factory for building chat completion request payloads."""

        @staticmethod
        def simple(
            message: str = "Hello, how are you?",
            model: str = TEST_MODEL_LLAMA,
            max_tokens: int = 100,
        ) -> dict[str, Any]:
            """Create a simple chat request."""
            return {
                "model": model,
                "messages": [
                    {"role": ROLE_USER, "content": message}
                ],
                "max_tokens": max_tokens,
                "stream": False,
            }

        @staticmethod
        def with_system(
            system: str,
            message: str,
            model: str = TEST_MODEL_LLAMA,
            max_tokens: int = 100,
        ) -> dict[str, Any]:
            """Create a chat request with system message."""
            return {
                "model": model,
                "messages": [
                    {"role": ROLE_SYSTEM, "content": system},
                    {"role": ROLE_USER, "content": message},
                ],
                "max_tokens": max_tokens,
                "stream": False,
            }

        @staticmethod
        def streaming(
            message: str = "Count from 1 to 5",
            model: str = TEST_MODEL_LLAMA,
            max_tokens: int = 100,
        ) -> dict[str, Any]:
            """Create a streaming chat request."""
            return {
                "model": model,
                "messages": [
                    {"role": ROLE_USER, "content": message}
                ],
                "max_tokens": max_tokens,
                "stream": True,
            }

        @staticmethod
        def orchestrated(
            message: str,
            mode: str = "critique",
            model: str = TEST_MODEL_PHI4,
            max_tokens: int = 200,
        ) -> dict[str, Any]:
            """Create an orchestrated chat request."""
            return {
                "model": model,
                "messages": [
                    {"role": ROLE_USER, "content": message}
                ],
                "max_tokens": max_tokens,
                "stream": False,
                "orchestration_mode": mode,
            }

        @staticmethod
        def code_task(
            message: str,
            model: str = TEST_MODEL_PHI4,
            max_tokens: int = 500,
        ) -> dict[str, Any]:
            """Create a code generation request."""
            return {
                "model": model,
                "messages": [
                    {"role": ROLE_SYSTEM, "content": "You are a helpful coding assistant."},
                    {"role": ROLE_USER, "content": message},
                ],
                "max_tokens": max_tokens,
                "stream": False,
                "task_type": "code",
            }

    return ChatRequestFactory


# =============================================================================
# Response Validators
# =============================================================================


@pytest.fixture
def response_validator() -> Any:
    """Validator for chat completion responses."""

    class ResponseValidator:
        """Validate chat completion response structure."""

        @staticmethod
        def validate_completion(response: dict[str, Any]) -> None:
            """Validate a non-streaming completion response."""
            assert "id" in response, "Response missing 'id'"
            assert response["id"].startswith("chatcmpl-"), "Invalid response ID format"
            assert response.get("object") == "chat.completion", "Invalid object type"
            assert "created" in response, "Response missing 'created'"
            assert "model" in response, "Response missing 'model'"
            assert "choices" in response, "Response missing 'choices'"
            assert len(response["choices"]) > 0, "No choices in response"

            choice = response["choices"][0]
            assert "index" in choice, "Choice missing 'index'"
            assert "message" in choice, "Choice missing 'message'"
            assert "finish_reason" in choice, "Choice missing 'finish_reason'"

            message = choice["message"]
            assert message.get("role") == "assistant", "Invalid message role"
            assert "content" in message, "Message missing 'content'"
            assert isinstance(message["content"], str), "Content must be string"

        @staticmethod
        def validate_usage(response: dict[str, Any]) -> None:
            """Validate usage statistics in response."""
            assert "usage" in response, "Response missing 'usage'"
            usage = response["usage"]
            assert "prompt_tokens" in usage, "Usage missing 'prompt_tokens'"
            assert "completion_tokens" in usage, "Usage missing 'completion_tokens'"
            assert "total_tokens" in usage, "Usage missing 'total_tokens'"
            assert usage["prompt_tokens"] > 0, "Prompt tokens should be > 0"
            assert usage["completion_tokens"] > 0, "Completion tokens should be > 0"
            assert usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]

        @staticmethod
        def validate_orchestration(
            response: dict[str, Any],
            expected_mode: str | None = None,
        ) -> None:
            """Validate orchestration metadata in response."""
            assert "orchestration" in response, "Response missing 'orchestration'"
            orch = response["orchestration"]
            assert "mode" in orch, "Orchestration missing 'mode'"
            assert "models_used" in orch, "Orchestration missing 'models_used'"

            if expected_mode:
                assert orch["mode"] == expected_mode, f"Expected mode {expected_mode}"

        @staticmethod
        def validate_streaming_chunk(chunk: dict[str, Any]) -> None:
            """Validate a streaming chunk."""
            assert "id" in chunk, "Chunk missing 'id'"
            assert "choices" in chunk, "Chunk missing 'choices'"
            assert len(chunk["choices"]) > 0, "No choices in chunk"

            choice = chunk["choices"][0]
            assert "index" in choice, "Choice missing 'index'"
            assert "delta" in choice, "Choice missing 'delta'"

    return ResponseValidator


# =============================================================================
# SSE Stream Parser
# =============================================================================


@pytest.fixture
def sse_parser() -> Any:
    """Parser for Server-Sent Events streams."""

    class SSEParser:
        """Parse SSE stream responses."""

        @staticmethod
        def _parse_data_content(data: str) -> dict[str, Any] | str | None:
            """Parse the data content from an SSE data line."""
            import json
            if data == SSE_DONE_MARKER:
                return SSE_DONE_MARKER
            try:
                result: dict[str, Any] = json.loads(data)
                return result
            except json.JSONDecodeError:
                return None

        @staticmethod
        def _is_ignorable_line(line: str) -> bool:
            """Check if line should be ignored."""
            return not line or line.startswith(":")

        @staticmethod
        def parse_line(line: str) -> dict[str, Any] | str | None:
            """Parse a single SSE line.

            Returns:
                - dict for JSON data
                - SSE_DONE_MARKER for stream end
                - None for empty/comment lines
            """
            line = line.strip()
            if SSEParser._is_ignorable_line(line):
                return None
            if line.startswith(SSE_DATA_PREFIX):
                data = line[len(SSE_DATA_PREFIX):]
                return SSEParser._parse_data_content(data)
            return None

        @staticmethod
        def parse_stream(content: str) -> list[dict[str, Any] | str]:
            """Parse entire SSE stream content."""
            results: list[dict[str, Any] | str] = []
            for line in content.split("\n"):
                parsed = SSEParser.parse_line(line)
                if parsed is not None:
                    results.append(parsed)
            return results

        @staticmethod
        def _extract_chunk_content(chunk: dict[str, Any]) -> str:
            """Extract content from a single chunk dict."""
            choices = chunk.get("choices", [])
            if choices:
                delta = choices[0].get("delta", {})
                return delta.get("content", "")
            return ""

        @staticmethod
        def extract_content(chunks: list[dict[str, Any] | str]) -> str:
            """Extract full content from streaming chunks."""
            content_parts: list[str] = []
            for chunk in chunks:
                if isinstance(chunk, dict):
                    content = SSEParser._extract_chunk_content(chunk)
                    if content:
                        content_parts.append(content)
            return "".join(content_parts)

    return SSEParser


# =============================================================================
# Test Data
# =============================================================================


@pytest.fixture
def sample_prompts() -> dict[str, str]:
    """Sample prompts for testing."""
    return {
        "simple": "Hello, how are you?",
        "code": "Write a Python function to calculate factorial.",
        "math": "What is 2 + 2?",
        "reasoning": "Explain why the sky is blue in simple terms.",
        "long": "Write a detailed essay about the history of computing." * 10,
    }


@pytest.fixture
def expected_models() -> list[str]:
    """List of expected available models."""
    return [
        TEST_MODEL_LLAMA,
        TEST_MODEL_PHI4,
        "deepseek-r1-7b",
        TEST_MODEL_QWEN,
    ]


# =============================================================================
# Model Discovery & Connectivity (Portable)
# =============================================================================


@pytest.fixture(scope="session")
async def available_models(async_client: httpx.AsyncClient) -> list[str]:
    """Discover available models from the API - works on any deployment."""
    try:
        response = await async_client.get("/v1/models")
        if response.status_code == 200:
            data = response.json()
            models = data.get("data", [])
            return [m.get("id", "") for m in models if m.get("id")]
    except httpx.ConnectError:
        pass
    return []


@pytest.fixture(scope="session")
async def loaded_models(async_client: httpx.AsyncClient) -> list[str]:
    """Get list of currently loaded models."""
    try:
        response = await async_client.get("/v1/models")
        if response.status_code == 200:
            data = response.json()
            models = data.get("data", [])
            return [
                m.get("id", "")
                for m in models
                if m.get("id") and m.get("status") == "loaded"
            ]
    except httpx.ConnectError:
        pass
    return []


# =============================================================================
# All Known Models for Connectivity Testing
# =============================================================================

ALL_MODELS = [
    {"id": "deepseek-r1-7b", "size_gb": 4.7, "purpose": "CoT Thinker"},
    {"id": TEST_MODEL_QWEN, "size_gb": 4.5, "purpose": "Coder"},
    {"id": "phi-4", "size_gb": 8.4, "purpose": "General"},
    {"id": "llama-3.2-3b", "size_gb": 2.0, "purpose": "Fast"},
    {"id": "phi-3-medium-128k", "size_gb": 8.6, "purpose": "Long Context"},
    {"id": "granite-8b-code-128k", "size_gb": 4.5, "purpose": "Code Analysis"},
    {"id": "qwen3-8b", "size_gb": 4.9, "purpose": "Balanced Reasoning"},
    {"id": "qwen3-coder-30b", "size_gb": 14.2, "purpose": "Advanced Coding"},
]


@pytest.fixture
def all_models() -> list[dict[str, Any]]:
    """Return all known models with metadata."""
    return ALL_MODELS


# =============================================================================
# D4 Preset Configuration
# =============================================================================

D4_PRESET = {
    "name": "Thinking + Code",
    "models": ["deepseek-r1-7b", TEST_MODEL_QWEN],
    "total_size_gb": 9.2,
    "orchestration_mode": "critique",
    "roles": {
        TEST_MODEL_QWEN: "generator",
        "deepseek-r1-7b": "critic",
    },
}


@pytest.fixture
def d4_preset() -> dict[str, Any]:
    """Return D4 preset configuration."""
    return D4_PRESET
