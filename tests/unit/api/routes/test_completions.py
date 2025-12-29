"""Unit tests for completions API routes.

Tests the /v1/chat/completions endpoint for OpenAI-compatible chat completions.

Reference: WBS-INF9 AC-9.1 through AC-9.5
"""

from __future__ import annotations

import json
import time
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI, status
from fastapi.testclient import TestClient


if TYPE_CHECKING:
    from src.models.responses import ChatCompletionChunk

# =============================================================================
# Constants (S1192: Avoid duplicated string literals)
# =============================================================================

COMPLETIONS_ENDPOINT = "/v1/chat/completions"
MODEL_PHI4 = "phi-4"
MODEL_DEEPSEEK = "deepseek-r1-7b"
ORCHESTRATION_MODE_SINGLE = "single"
ORCHESTRATION_MODE_CRITIQUE = "critique"
ROLE_ASSISTANT = "assistant"
ROLE_USER = "user"
ROLE_SYSTEM = "system"
OBJECT_CHAT_COMPLETION = "chat.completion"
OBJECT_CHAT_COMPLETION_CHUNK = "chat.completion.chunk"
FINISH_REASON_STOP = "stop"
SSE_DONE = "data: [DONE]"


# =============================================================================
# TestCompletionsRouterImport
# =============================================================================


class TestCompletionsRouterImport:
    """Test that completions router can be imported."""

    def test_completions_router_importable(self) -> None:
        """AC-9.1: Completions router exists and can be imported."""
        from src.api.routes.completions import router

        assert router is not None

    def test_create_chat_completion_importable(self) -> None:
        """AC-9.1: Chat completion endpoint function is importable."""
        from src.api.routes.completions import create_chat_completion

        assert create_chat_completion is not None


# =============================================================================
# Test Fixtures
# =============================================================================


def _create_mock_response(
    model: str = MODEL_PHI4,
    content: str = "This is a test response.",
    prompt_tokens: int = 25,
    completion_tokens: int = 10,
) -> MagicMock:
    """Create a mock ChatCompletionResponse."""
    response = MagicMock()
    response.id = "chatcmpl-test123"
    response.object = OBJECT_CHAT_COMPLETION
    response.created = int(time.time())
    response.model = model

    # Create choice
    choice = MagicMock()
    choice.index = 0
    choice.message = MagicMock()
    choice.message.role = ROLE_ASSISTANT
    choice.message.content = content
    choice.message.tool_calls = None
    choice.finish_reason = FINISH_REASON_STOP
    response.choices = [choice]

    # Create usage
    response.usage = MagicMock()
    response.usage.prompt_tokens = prompt_tokens
    response.usage.completion_tokens = completion_tokens
    response.usage.total_tokens = prompt_tokens + completion_tokens

    # Orchestration metadata
    response.orchestration = None

    # For JSON serialization
    response.model_dump.return_value = {
        "id": response.id,
        "object": response.object,
        "created": response.created,
        "model": response.model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": ROLE_ASSISTANT,
                    "content": content,
                    "tool_calls": None,
                },
                "finish_reason": FINISH_REASON_STOP,
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
        "orchestration": None,
    }

    return response


def _create_mock_chunk(
    chunk_id: str,
    model: str,
    content: str | None = None,
    role: str | None = None,
    finish_reason: str | None = None,
) -> MagicMock:
    """Create a mock ChatCompletionChunk."""
    chunk = MagicMock()
    chunk.id = chunk_id
    chunk.object = OBJECT_CHAT_COMPLETION_CHUNK
    chunk.created = int(time.time())
    chunk.model = model

    delta = MagicMock()
    delta.role = role
    delta.content = content
    delta.tool_calls = None

    choice = MagicMock()
    choice.index = 0
    choice.delta = delta
    choice.finish_reason = finish_reason

    chunk.choices = [choice]

    # For SSE serialization
    chunk_data = {
        "id": chunk_id,
        "object": OBJECT_CHAT_COMPLETION_CHUNK,
        "created": chunk.created,
        "model": model,
        "choices": [
            {
                "index": 0,
                "delta": {"role": role, "content": content},
                "finish_reason": finish_reason,
            }
        ],
    }
    chunk.model_dump_json.return_value = json.dumps(chunk_data)
    chunk.to_sse.return_value = f"data: {json.dumps(chunk_data)}"

    return chunk


@pytest.fixture
def mock_provider() -> MagicMock:
    """Create mock InferenceProvider."""
    provider = MagicMock()
    provider.model_info = MagicMock()
    provider.model_info.model_id = MODEL_PHI4
    provider.model_info.context_length = 16384
    provider.is_loaded = True
    provider.generate = AsyncMock(return_value=_create_mock_response())
    return provider


@pytest.fixture
def mock_model_manager(mock_provider: MagicMock) -> MagicMock:
    """Create mock ModelManager."""
    manager = MagicMock()
    # get_provider is async, so use AsyncMock
    manager.get_provider = AsyncMock(return_value=mock_provider)
    manager.get_loaded_models.return_value = [MODEL_PHI4]
    return manager


@pytest.fixture
def app_with_manager(mock_model_manager: MagicMock) -> FastAPI:
    """Create FastAPI app with mock model manager."""
    from src.api.routes.completions import router

    app = FastAPI()
    app.include_router(router, prefix="/v1")
    app.state.model_manager = mock_model_manager
    return app


@pytest.fixture
def client(app_with_manager: FastAPI) -> TestClient:
    """Create test client."""
    return TestClient(app_with_manager)


@pytest.fixture
def valid_request() -> dict[str, Any]:
    """Create a valid chat completion request."""
    return {
        "model": MODEL_PHI4,
        "messages": [
            {"role": ROLE_SYSTEM, "content": "You are a helpful assistant."},
            {"role": ROLE_USER, "content": "Hello, how are you?"},
        ],
        "stream": False,
        "max_tokens": 100,
        "temperature": 0.7,
    }


# =============================================================================
# TestNonStreamingCompletion (AC-9.1, AC-9.2, AC-9.4)
# =============================================================================


class TestNonStreamingCompletion:
    """Test non-streaming /v1/chat/completions endpoint."""

    def test_completion_returns_200(
        self, client: TestClient, valid_request: dict[str, Any]
    ) -> None:
        """AC-9.1: POST /v1/chat/completions returns 200."""
        response = client.post(COMPLETIONS_ENDPOINT, json=valid_request)

        assert response.status_code == status.HTTP_200_OK

    def test_completion_returns_valid_json(
        self, client: TestClient, valid_request: dict[str, Any]
    ) -> None:
        """AC-9.2: Response is valid JSON."""
        response = client.post(COMPLETIONS_ENDPOINT, json=valid_request)
        data = response.json()

        assert isinstance(data, dict)

    def test_completion_has_required_fields(
        self, client: TestClient, valid_request: dict[str, Any]
    ) -> None:
        """AC-9.2: Response has required OpenAI fields."""
        response = client.post(COMPLETIONS_ENDPOINT, json=valid_request)
        data = response.json()

        assert "id" in data
        assert "object" in data
        assert "created" in data
        assert "model" in data
        assert "choices" in data
        assert "usage" in data

    def test_completion_object_is_chat_completion(
        self, client: TestClient, valid_request: dict[str, Any]
    ) -> None:
        """AC-9.2: Response object is 'chat.completion'."""
        response = client.post(COMPLETIONS_ENDPOINT, json=valid_request)
        data = response.json()

        assert data["object"] == OBJECT_CHAT_COMPLETION

    def test_completion_has_choices_array(
        self, client: TestClient, valid_request: dict[str, Any]
    ) -> None:
        """AC-9.2: Response has choices array."""
        response = client.post(COMPLETIONS_ENDPOINT, json=valid_request)
        data = response.json()

        assert isinstance(data["choices"], list)
        assert len(data["choices"]) >= 1

    def test_choice_has_message(
        self, client: TestClient, valid_request: dict[str, Any]
    ) -> None:
        """AC-9.2: Choice has message with role and content."""
        response = client.post(COMPLETIONS_ENDPOINT, json=valid_request)
        data = response.json()

        choice = data["choices"][0]
        assert "message" in choice
        assert choice["message"]["role"] == ROLE_ASSISTANT
        assert "content" in choice["message"]

    def test_choice_has_finish_reason(
        self, client: TestClient, valid_request: dict[str, Any]
    ) -> None:
        """AC-9.2: Choice has finish_reason."""
        response = client.post(COMPLETIONS_ENDPOINT, json=valid_request)
        data = response.json()

        choice = data["choices"][0]
        assert "finish_reason" in choice

    def test_completion_includes_usage(
        self, client: TestClient, valid_request: dict[str, Any]
    ) -> None:
        """AC-9.4: Response includes usage statistics."""
        response = client.post(COMPLETIONS_ENDPOINT, json=valid_request)
        data = response.json()

        assert "usage" in data
        usage = data["usage"]
        assert "prompt_tokens" in usage
        assert "completion_tokens" in usage
        assert "total_tokens" in usage

    def test_usage_total_is_sum(
        self, client: TestClient, valid_request: dict[str, Any]
    ) -> None:
        """AC-9.4: Usage total equals prompt + completion."""
        response = client.post(COMPLETIONS_ENDPOINT, json=valid_request)
        data = response.json()

        usage = data["usage"]
        assert usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]

    def test_completion_model_matches_request(
        self, client: TestClient, valid_request: dict[str, Any]
    ) -> None:
        """AC-9.2: Response model matches request model."""
        response = client.post(COMPLETIONS_ENDPOINT, json=valid_request)
        data = response.json()

        assert data["model"] == valid_request["model"]


# =============================================================================
# TestRequestValidation (AC-9.1)
# =============================================================================


class TestRequestValidation:
    """Test request validation for /v1/chat/completions."""

    def test_missing_model_returns_422(self, client: TestClient) -> None:
        """AC-9.1: Missing model returns 422."""
        request = {
            "messages": [{"role": ROLE_USER, "content": "Hello"}],
        }
        response = client.post(COMPLETIONS_ENDPOINT, json=request)

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_missing_messages_returns_422(self, client: TestClient) -> None:
        """AC-9.1: Missing messages returns 422."""
        request = {
            "model": MODEL_PHI4,
        }
        response = client.post(COMPLETIONS_ENDPOINT, json=request)

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_empty_messages_returns_422(self, client: TestClient) -> None:
        """AC-9.1: Empty messages returns 422."""
        request = {
            "model": MODEL_PHI4,
            "messages": [],
        }
        response = client.post(COMPLETIONS_ENDPOINT, json=request)

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_invalid_temperature_returns_422(self, client: TestClient) -> None:
        """AC-9.1: Temperature > 2 returns 422."""
        request = {
            "model": MODEL_PHI4,
            "messages": [{"role": ROLE_USER, "content": "Hello"}],
            "temperature": 3.0,
        }
        response = client.post(COMPLETIONS_ENDPOINT, json=request)

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_valid_extension_fields_accepted(
        self, client: TestClient, valid_request: dict[str, Any]
    ) -> None:
        """AC-9.1: Extension fields (orchestration_mode, etc.) accepted."""
        valid_request["orchestration_mode"] = ORCHESTRATION_MODE_CRITIQUE
        valid_request["task_type"] = "code"
        valid_request["priority"] = 3

        response = client.post(COMPLETIONS_ENDPOINT, json=valid_request)

        assert response.status_code == status.HTTP_200_OK


# =============================================================================
# TestModelNotFound
# =============================================================================


class TestModelNotFound:
    """Test model not found handling."""

    def test_model_not_loaded_returns_404(
        self, client: TestClient, mock_model_manager: MagicMock, valid_request: dict[str, Any]
    ) -> None:
        """Returns 404 when model not loaded."""
        from src.services.model_manager import ModelNotLoadedError

        mock_model_manager.get_provider.side_effect = ModelNotLoadedError(
            f"Model '{MODEL_PHI4}' is not loaded"
        )

        response = client.post(COMPLETIONS_ENDPOINT, json=valid_request)

        assert response.status_code == status.HTTP_404_NOT_FOUND

    def test_404_includes_error_message(
        self, client: TestClient, mock_model_manager: MagicMock, valid_request: dict[str, Any]
    ) -> None:
        """404 response includes clear error message."""
        from src.services.model_manager import ModelNotLoadedError

        mock_model_manager.get_provider.side_effect = ModelNotLoadedError(
            f"Model '{MODEL_PHI4}' is not loaded"
        )

        response = client.post(COMPLETIONS_ENDPOINT, json=valid_request)
        data = response.json()

        assert "detail" in data
        assert MODEL_PHI4 in data["detail"]


# =============================================================================
# TestNoModelManager
# =============================================================================


class TestNoModelManager:
    """Test when model manager not initialized."""

    @pytest.fixture
    def app_no_manager(self) -> FastAPI:
        """Create FastAPI app without model manager."""
        from src.api.routes.completions import router

        app = FastAPI()
        app.include_router(router, prefix="/v1")
        return app

    @pytest.fixture
    def client_no_manager(self, app_no_manager: FastAPI) -> TestClient:
        """Create test client without model manager."""
        return TestClient(app_no_manager)

    def test_returns_503_when_no_manager(
        self, client_no_manager: TestClient, valid_request: dict[str, Any]
    ) -> None:
        """Returns 503 when model manager not initialized."""
        response = client_no_manager.post(COMPLETIONS_ENDPOINT, json=valid_request)

        assert response.status_code == status.HTTP_503_SERVICE_UNAVAILABLE


# =============================================================================
# TestStreamingCompletion (AC-9.3)
# =============================================================================


class TestStreamingCompletion:
    """Test streaming /v1/chat/completions endpoint."""

    @pytest.fixture
    def streaming_request(self, valid_request: dict[str, Any]) -> dict[str, Any]:
        """Create streaming request."""
        valid_request["stream"] = True
        return valid_request

    @pytest.fixture
    def mock_stream_provider(self) -> MagicMock:
        """Create mock provider with streaming support."""
        provider = MagicMock()
        provider.model_info = MagicMock()
        provider.model_info.model_id = MODEL_PHI4
        provider.is_loaded = True

        async def mock_stream(
            request: object,
        ) -> AsyncIterator[ChatCompletionChunk]:
            """Mock streaming generator."""
            chunk_id = "chatcmpl-stream123"

            # First chunk with role
            yield _create_mock_chunk(chunk_id, MODEL_PHI4, role=ROLE_ASSISTANT)

            # Content chunks
            for token in ["Hello", " ", "world", "!"]:
                yield _create_mock_chunk(chunk_id, MODEL_PHI4, content=token)

            # Final chunk with finish_reason
            yield _create_mock_chunk(
                chunk_id, MODEL_PHI4, finish_reason=FINISH_REASON_STOP
            )

        provider.stream = mock_stream
        return provider

    @pytest.fixture
    def app_streaming(self, mock_stream_provider: MagicMock) -> FastAPI:
        """Create app with streaming provider."""
        from src.api.routes.completions import router

        manager = MagicMock()
        manager.get_provider = AsyncMock(return_value=mock_stream_provider)
        manager.get_loaded_models.return_value = [MODEL_PHI4]

        app = FastAPI()
        app.include_router(router, prefix="/v1")
        app.state.model_manager = manager
        return app

    @pytest.fixture
    def client_streaming(self, app_streaming: FastAPI) -> TestClient:
        """Create client for streaming tests."""
        return TestClient(app_streaming)

    def test_streaming_returns_200(
        self, client_streaming: TestClient, streaming_request: dict[str, Any]
    ) -> None:
        """AC-9.3: Streaming request returns 200."""
        response = client_streaming.post(
            COMPLETIONS_ENDPOINT, json=streaming_request
        )

        assert response.status_code == status.HTTP_200_OK

    def test_streaming_content_type(
        self, client_streaming: TestClient, streaming_request: dict[str, Any]
    ) -> None:
        """AC-9.3: Streaming response has text/event-stream content type."""
        response = client_streaming.post(
            COMPLETIONS_ENDPOINT, json=streaming_request
        )

        assert "text/event-stream" in response.headers.get("content-type", "")

    def test_streaming_has_sse_format(
        self, client_streaming: TestClient, streaming_request: dict[str, Any]
    ) -> None:
        """AC-9.3: Streaming response uses SSE format (data: prefix)."""
        response = client_streaming.post(
            COMPLETIONS_ENDPOINT, json=streaming_request
        )

        lines = response.text.strip().split("\n")
        data_lines = [line for line in lines if line.startswith("data:")]
        assert len(data_lines) >= 1

    def test_streaming_ends_with_done(
        self, client_streaming: TestClient, streaming_request: dict[str, Any]
    ) -> None:
        """AC-9.3: Streaming ends with data: [DONE]."""
        response = client_streaming.post(
            COMPLETIONS_ENDPOINT, json=streaming_request
        )

        assert SSE_DONE in response.text

    def test_streaming_chunks_are_valid_json(
        self, client_streaming: TestClient, streaming_request: dict[str, Any]
    ) -> None:
        """AC-9.3: Each SSE chunk (except [DONE]) is valid JSON."""
        response = client_streaming.post(
            COMPLETIONS_ENDPOINT, json=streaming_request
        )

        lines = response.text.strip().split("\n")
        for line in lines:
            if line.startswith("data:") and "[DONE]" not in line:
                json_str = line[5:].strip()  # Remove "data: " prefix
                data = json.loads(json_str)  # Should not raise
                assert "id" in data
                assert "choices" in data

    def test_streaming_chunk_object_type(
        self, client_streaming: TestClient, streaming_request: dict[str, Any]
    ) -> None:
        """AC-9.3: Streaming chunk has object type 'chat.completion.chunk'."""
        response = client_streaming.post(
            COMPLETIONS_ENDPOINT, json=streaming_request
        )

        lines = response.text.strip().split("\n")
        for line in lines:
            if line.startswith("data:") and "[DONE]" not in line:
                json_str = line[5:].strip()
                data = json.loads(json_str)
                assert data["object"] == OBJECT_CHAT_COMPLETION_CHUNK


# =============================================================================
# TestOrchestrationMetadata (AC-9.5)
# =============================================================================


class TestOrchestrationMetadata:
    """Test orchestration metadata in responses."""

    @pytest.fixture
    def mock_response_with_orchestration(self) -> MagicMock:
        """Create mock response with orchestration metadata."""
        response = _create_mock_response()

        orchestration = {
            "mode": ORCHESTRATION_MODE_CRITIQUE,
            "models_used": [MODEL_PHI4, MODEL_DEEPSEEK],
            "total_inference_time_ms": 1500.0,
            "rounds": 2,
            "final_score": 0.92,
            "agreement_score": None,
        }

        response.orchestration = MagicMock()
        response.orchestration.mode = ORCHESTRATION_MODE_CRITIQUE
        response.orchestration.models_used = [MODEL_PHI4, MODEL_DEEPSEEK]
        response.orchestration.total_inference_time_ms = 1500.0
        response.orchestration.rounds = 2
        response.orchestration.final_score = 0.92
        response.orchestration.agreement_score = None

        # Update model_dump to include orchestration
        dump = response.model_dump.return_value
        dump["orchestration"] = orchestration
        response.model_dump.return_value = dump

        return response

    @pytest.fixture
    def app_with_orchestration(
        self, mock_response_with_orchestration: MagicMock
    ) -> FastAPI:
        """Create app that returns orchestration metadata."""
        from src.api.routes.completions import router

        provider = MagicMock()
        provider.model_info = MagicMock()
        provider.model_info.model_id = MODEL_PHI4
        provider.is_loaded = True
        provider.generate = AsyncMock(return_value=mock_response_with_orchestration)

        manager = MagicMock()
        manager.get_provider = AsyncMock(return_value=provider)
        manager.get_loaded_models.return_value = [MODEL_PHI4, MODEL_DEEPSEEK]

        app = FastAPI()
        app.include_router(router, prefix="/v1")
        app.state.model_manager = manager
        return app

    @pytest.fixture
    def client_orchestration(self, app_with_orchestration: FastAPI) -> TestClient:
        """Create client for orchestration tests."""
        return TestClient(app_with_orchestration)

    def test_response_includes_orchestration(
        self, client_orchestration: TestClient, valid_request: dict[str, Any]
    ) -> None:
        """AC-9.5: Response includes orchestration metadata."""
        valid_request["orchestration_mode"] = ORCHESTRATION_MODE_CRITIQUE

        response = client_orchestration.post(COMPLETIONS_ENDPOINT, json=valid_request)
        data = response.json()

        assert "orchestration" in data

    def test_orchestration_has_mode(
        self, client_orchestration: TestClient, valid_request: dict[str, Any]
    ) -> None:
        """AC-9.5: Orchestration includes mode."""
        valid_request["orchestration_mode"] = ORCHESTRATION_MODE_CRITIQUE

        response = client_orchestration.post(COMPLETIONS_ENDPOINT, json=valid_request)
        data = response.json()

        assert data["orchestration"]["mode"] == ORCHESTRATION_MODE_CRITIQUE

    def test_orchestration_has_models_used(
        self, client_orchestration: TestClient, valid_request: dict[str, Any]
    ) -> None:
        """AC-9.5: Orchestration includes models_used list."""
        valid_request["orchestration_mode"] = ORCHESTRATION_MODE_CRITIQUE

        response = client_orchestration.post(COMPLETIONS_ENDPOINT, json=valid_request)
        data = response.json()

        assert "models_used" in data["orchestration"]
        assert isinstance(data["orchestration"]["models_used"], list)
        assert MODEL_PHI4 in data["orchestration"]["models_used"]

    def test_orchestration_has_rounds(
        self, client_orchestration: TestClient, valid_request: dict[str, Any]
    ) -> None:
        """AC-9.5: Orchestration includes rounds (for critique mode)."""
        valid_request["orchestration_mode"] = ORCHESTRATION_MODE_CRITIQUE

        response = client_orchestration.post(COMPLETIONS_ENDPOINT, json=valid_request)
        data = response.json()

        assert "rounds" in data["orchestration"]
        assert data["orchestration"]["rounds"] == 2

    def test_orchestration_has_confidence(
        self, client_orchestration: TestClient, valid_request: dict[str, Any]
    ) -> None:
        """AC-9.5: Orchestration includes confidence/score."""
        valid_request["orchestration_mode"] = ORCHESTRATION_MODE_CRITIQUE

        response = client_orchestration.post(COMPLETIONS_ENDPOINT, json=valid_request)
        data = response.json()

        # Can be final_score or agreement_score depending on mode
        assert (
            "final_score" in data["orchestration"]
            or "agreement_score" in data["orchestration"]
        )


# =============================================================================
# TestInferenceErrors
# =============================================================================


class TestInferenceErrors:
    """Test inference error handling."""

    def test_inference_error_returns_500(
        self, client: TestClient, mock_provider: MagicMock, valid_request: dict[str, Any]
    ) -> None:
        """Inference error returns 500."""
        mock_provider.generate.side_effect = RuntimeError("Inference failed")

        response = client.post(COMPLETIONS_ENDPOINT, json=valid_request)

        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR

    def test_500_includes_error_detail(
        self, client: TestClient, mock_provider: MagicMock, valid_request: dict[str, Any]
    ) -> None:
        """500 response includes error detail."""
        mock_provider.generate.side_effect = RuntimeError("Model crashed")

        response = client.post(COMPLETIONS_ENDPOINT, json=valid_request)
        data = response.json()

        assert "detail" in data
