"""Unit tests for Phase A usage tracking in the completions route.

Tests that _record_usage() is called on every inference request
(both streaming and non-streaming paths).

Reference: WBS-MESH-A Task A.3
"""

from __future__ import annotations

import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI, status
from fastapi.testclient import TestClient


# =============================================================================
# Constants
# =============================================================================

COMPLETIONS_ENDPOINT = "/v1/chat/completions"
MODEL_PHI4 = "phi-4"


# =============================================================================
# Fixtures
# =============================================================================


def _create_mock_response(model: str = MODEL_PHI4) -> MagicMock:
    """Create a mock ChatCompletionResponse."""
    response = MagicMock()
    response.id = "chatcmpl-test123"
    response.object = "chat.completion"
    response.created = int(time.time())
    response.model = model
    response.choices = [
        MagicMock(
            index=0,
            message=MagicMock(role="assistant", content="Hello"),
            finish_reason="stop",
        )
    ]
    response.usage = MagicMock(
        prompt_tokens=10, completion_tokens=5, total_tokens=15
    )
    response.orchestration = None
    response.model_dump = MagicMock(
        return_value={
            "id": response.id,
            "object": response.object,
            "model": model,
            "choices": [],
            "usage": {},
        }
    )
    return response


def _create_test_app() -> FastAPI:
    """Create a FastAPI app with completions router and mock model manager."""
    from src.api.routes.completions import router

    app = FastAPI()
    app.include_router(router, prefix="/v1")

    # Wire mock model manager
    mock_manager = MagicMock()
    mock_provider = MagicMock()
    mock_provider.generate = AsyncMock(return_value=_create_mock_response())
    mock_manager.get_provider = MagicMock(return_value=mock_provider)
    app.state.model_manager = mock_manager
    app.state.queue_manager = None

    return app


@pytest.fixture
def client() -> TestClient:
    """Create a test client with completions route."""
    app = _create_test_app()
    return TestClient(app)


@pytest.fixture
def sample_request() -> dict:
    """Minimal chat completion request."""
    return {
        "model": MODEL_PHI4,
        "messages": [{"role": "user", "content": "Hello"}],
        "stream": False,
    }


# =============================================================================
# TestUsageTrackingNonStreaming
# =============================================================================


class TestUsageTrackingNonStreaming:
    """Test that _record_usage is called on non-streaming requests."""

    @patch("src.api.routes.completions.get_config_publisher")
    def test_record_usage_called_on_non_streaming(
        self, mock_get_publisher: MagicMock, client: TestClient, sample_request: dict
    ) -> None:
        """AC-A.1: Non-streaming inference request calls record_model_usage."""
        mock_publisher = AsyncMock()
        mock_publisher.record_model_usage = AsyncMock(return_value=True)
        mock_get_publisher.return_value = mock_publisher

        response = client.post(COMPLETIONS_ENDPOINT, json=sample_request)

        assert response.status_code == status.HTTP_200_OK
        mock_publisher.record_model_usage.assert_awaited_once_with(MODEL_PHI4)

    @patch("src.api.routes.completions.get_config_publisher")
    def test_inference_succeeds_when_publisher_is_none(
        self, mock_get_publisher: MagicMock, client: TestClient, sample_request: dict
    ) -> None:
        """AC-A.4: Inference succeeds even when publisher is None."""
        mock_get_publisher.return_value = None

        response = client.post(COMPLETIONS_ENDPOINT, json=sample_request)

        assert response.status_code == status.HTTP_200_OK

    @patch("src.api.routes.completions.get_config_publisher")
    def test_inference_succeeds_when_usage_tracking_fails(
        self, mock_get_publisher: MagicMock, client: TestClient, sample_request: dict
    ) -> None:
        """AC-A.4: Inference succeeds even when record_model_usage returns False."""
        mock_publisher = AsyncMock()
        mock_publisher.record_model_usage = AsyncMock(return_value=False)
        mock_get_publisher.return_value = mock_publisher

        response = client.post(COMPLETIONS_ENDPOINT, json=sample_request)

        assert response.status_code == status.HTTP_200_OK


# =============================================================================
# TestUsageTrackingStreaming
# =============================================================================


class TestUsageTrackingStreaming:
    """Test that _record_usage is called on streaming requests."""

    @patch("src.api.routes.completions.get_config_publisher")
    def test_record_usage_called_on_streaming(
        self, mock_get_publisher: MagicMock, client: TestClient
    ) -> None:
        """AC-A.1: Streaming inference request calls record_model_usage."""
        mock_publisher = AsyncMock()
        mock_publisher.record_model_usage = AsyncMock(return_value=True)
        mock_get_publisher.return_value = mock_publisher

        # Override the provider to support streaming
        app = client.app
        mock_provider = app.state.model_manager.get_provider()

        async def mock_stream(_req: object) -> list:
            """Empty stream generator."""
            return []

        mock_provider.stream = mock_stream

        streaming_request = {
            "model": MODEL_PHI4,
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": True,
        }

        response = client.post(COMPLETIONS_ENDPOINT, json=streaming_request)

        assert response.status_code == status.HTTP_200_OK
        mock_publisher.record_model_usage.assert_awaited_once_with(MODEL_PHI4)
