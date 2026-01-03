"""pytest configuration and fixtures for inference-service tests.

This module provides shared fixtures for unit and integration tests.
Following TDD principles, fixtures are minimal and focused.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator, Generator
from typing import TYPE_CHECKING

import pytest
from httpx import ASGITransport, AsyncClient


if TYPE_CHECKING:
    from fastapi import FastAPI

# =============================================================================
# Constants (S1192: Avoid duplicated string literals)
# =============================================================================

OBJECT_CHAT_COMPLETION = "chat.completion"
OBJECT_CHAT_COMPLETION_CHUNK = "chat.completion.chunk"
TEST_MODEL_ID = "phi-4"
TEST_COMPLETION_ID = "chatcmpl-test123"


# =============================================================================
# Markers
# =============================================================================


def pytest_configure(config: pytest.Config) -> None:
    """Register custom pytest markers."""
    config.addinivalue_line("markers", "unit: Unit tests (fast, no external dependencies)")
    config.addinivalue_line("markers", "integration: Integration tests (may require models)")
    config.addinivalue_line("markers", "slow: Slow tests (model loading, large inputs)")


# =============================================================================
# Environment Fixtures
# =============================================================================


@pytest.fixture(scope="session")
def test_env_vars() -> dict[str, str]:
    """Provide test environment variables.

    Returns:
        Dictionary of environment variables for testing.
    """
    # Using relative paths - pytest tmpdir fixtures handle actual temp directories
    # These are placeholder values that get overridden by test-specific tmpdir
    return {
        "INFERENCE_PORT": "8085",
        "INFERENCE_HOST": "127.0.0.1",
        "INFERENCE_LOG_LEVEL": "DEBUG",
        "INFERENCE_MODELS_DIR": "test-models",
        "INFERENCE_CONFIG_DIR": "test-config",
        "INFERENCE_ORCHESTRATION_MODE": "single",
        "INFERENCE_SKIP_PATH_VALIDATION": "true",  # Skip path checks in tests
    }


@pytest.fixture
def mock_env(test_env_vars: dict[str, str], monkeypatch: pytest.MonkeyPatch) -> None:
    """Set test environment variables.

    Args:
        test_env_vars: Test environment variables.
        monkeypatch: pytest monkeypatch fixture.
    """
    for key, value in test_env_vars.items():
        monkeypatch.setenv(key, value)


# =============================================================================
# App Fixtures
# =============================================================================


@pytest.fixture
def app() -> Generator[FastAPI, None, None]:
    """Create a test FastAPI application.

    Yields:
        FastAPI application instance for testing.

    Note:
        This fixture will be expanded in WBS-INF2 when main.py is implemented.
    """
    # Placeholder until src.main is implemented (WBS-INF2)
    from fastapi import FastAPI

    test_app = FastAPI(title="inference-service-test")

    @test_app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    yield test_app


@pytest.fixture
async def async_client(app: FastAPI) -> AsyncGenerator[AsyncClient, None]:
    """Create an async HTTP client for testing.

    Args:
        app: FastAPI application.

    Yields:
        AsyncClient for making test requests.
    """
    transport = ASGITransport(app=app)
    # Using https scheme for test transport (security best practice)
    async with AsyncClient(transport=transport, base_url="https://testserver") as client:
        yield client


# =============================================================================
# Model Fixtures (Stubs - expanded in WBS-INF5)
# =============================================================================


@pytest.fixture
def mock_model_response() -> dict[str, object]:
    """Provide a mock model response.

    Returns:
        Dictionary mimicking ChatCompletionResponse structure.
    """
    return {
        "id": TEST_COMPLETION_ID,
        "object": OBJECT_CHAT_COMPLETION,
        "created": 1234567890,
        "model": TEST_MODEL_ID,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "This is a test response.",
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 20,
            "total_tokens": 30,
        },
    }


@pytest.fixture
def mock_streaming_chunks() -> list[dict[str, object]]:
    """Provide mock streaming response chunks.

    Returns:
        List of dictionaries mimicking ChatCompletionChunk structure.
    """
    return [
        {
            "id": TEST_COMPLETION_ID,
            "object": OBJECT_CHAT_COMPLETION_CHUNK,
            "created": 1234567890,
            "model": TEST_MODEL_ID,
            "choices": [
                {
                    "index": 0,
                    "delta": {"role": "assistant"},
                    "finish_reason": None,
                }
            ],
        },
        {
            "id": TEST_COMPLETION_ID,
            "object": OBJECT_CHAT_COMPLETION_CHUNK,
            "created": 1234567890,
            "model": TEST_MODEL_ID,
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": "Hello"},
                    "finish_reason": None,
                }
            ],
        },
        {
            "id": TEST_COMPLETION_ID,
            "object": OBJECT_CHAT_COMPLETION_CHUNK,
            "created": 1234567890,
            "model": TEST_MODEL_ID,
            "choices": [
                {
                    "index": 0,
                    "delta": {},
                    "finish_reason": "stop",
                }
            ],
        },
    ]


# =============================================================================
# Request Fixtures
# =============================================================================


@pytest.fixture
def sample_chat_request() -> dict[str, object]:
    """Provide a sample chat completion request.

    Returns:
        Dictionary mimicking ChatCompletionRequest structure.
    """
    return {
        "model": TEST_MODEL_ID,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello, how are you?"},
        ],
        "max_tokens": 100,
        "temperature": 0.7,
        "stream": False,
    }


@pytest.fixture
def sample_streaming_request(sample_chat_request: dict[str, object]) -> dict[str, object]:
    """Provide a sample streaming chat request.

    Args:
        sample_chat_request: Base chat request.

    Returns:
        Dictionary with stream=True.
    """
    return {**sample_chat_request, "stream": True}
