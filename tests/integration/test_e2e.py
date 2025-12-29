"""End-to-end integration tests for inference-service.

Tests the complete request → response flow with actual model inference.

AC-21.1: End-to-end test: request → response with llama-3.2-3b
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest


if TYPE_CHECKING:
    import httpx


pytestmark = [pytest.mark.integration, pytest.mark.asyncio]


# =============================================================================
# Health Check Tests
# =============================================================================


class TestServiceHealth:
    """Test service health endpoints."""

    async def test_health_endpoint_returns_200(
        self,
        client: httpx.AsyncClient,
        skip_if_service_unavailable: None,
    ) -> None:
        """Test /health endpoint returns 200 when service is up."""
        response = await client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data.get("status") == "ok"

    async def test_health_ready_endpoint(
        self,
        client: httpx.AsyncClient,
        skip_if_service_unavailable: None,
    ) -> None:
        """Test /health/ready endpoint returns service readiness."""
        response = await client.get("/health/ready")

        # Either 200 (ready) or 503 (not ready) are valid responses
        assert response.status_code in [200, 503]
        data = response.json()
        assert "status" in data


# =============================================================================
# Models Endpoint Tests
# =============================================================================


class TestModelsEndpoint:
    """Test /v1/models endpoint."""

    async def test_list_models_returns_data(
        self,
        client: httpx.AsyncClient,
        skip_if_service_unavailable: None,
    ) -> None:
        """Test GET /v1/models returns model list."""
        response = await client.get("/v1/models")

        assert response.status_code == 200
        data = response.json()
        assert "data" in data
        assert isinstance(data["data"], list)

    async def test_model_info_structure(
        self,
        client: httpx.AsyncClient,
        skip_if_service_unavailable: None,
    ) -> None:
        """Test model info contains required fields."""
        response = await client.get("/v1/models")

        assert response.status_code == 200
        data = response.json()

        if data["data"]:  # If any models available
            model = data["data"][0]
            assert "id" in model
            assert "status" in model


# =============================================================================
# Chat Completion E2E Tests
# =============================================================================


class TestChatCompletionE2E:
    """End-to-end tests for chat completion."""

    @pytest.mark.requires_model
    async def test_simple_completion(
        self,
        client: httpx.AsyncClient,
        skip_if_no_models: None,
        chat_request_factory: Any,
        response_validator: Any,
    ) -> None:
        """Test simple chat completion request → response.

        AC-21.1: End-to-end test with llama-3.2-3b (or available model).
        """
        request = chat_request_factory.simple(
            message="What is 2 + 2? Answer with just the number.",
            max_tokens=50,
        )

        response = await client.post("/v1/chat/completions", json=request)

        assert response.status_code == 200
        data = response.json()

        # Validate response structure
        response_validator.validate_completion(data)

        # Check we got content
        content = data["choices"][0]["message"]["content"]
        assert len(content) > 0

    @pytest.mark.requires_model
    async def test_completion_with_system_message(
        self,
        client: httpx.AsyncClient,
        skip_if_no_models: None,
        chat_request_factory: Any,
        response_validator: Any,
    ) -> None:
        """Test completion with system message."""
        request = chat_request_factory.with_system(
            system="You are a helpful assistant. Always be brief.",
            message="Say hello.",
            max_tokens=50,
        )

        response = await client.post("/v1/chat/completions", json=request)

        assert response.status_code == 200
        data = response.json()
        response_validator.validate_completion(data)

    @pytest.mark.requires_model
    async def test_completion_includes_usage(
        self,
        client: httpx.AsyncClient,
        skip_if_no_models: None,
        chat_request_factory: Any,
        response_validator: Any,
    ) -> None:
        """Test completion response includes usage statistics."""
        request = chat_request_factory.simple(
            message="Hello",
            max_tokens=20,
        )

        response = await client.post("/v1/chat/completions", json=request)

        assert response.status_code == 200
        data = response.json()
        response_validator.validate_usage(data)

    @pytest.mark.requires_model
    async def test_completion_respects_max_tokens(
        self,
        client: httpx.AsyncClient,
        skip_if_no_models: None,
        chat_request_factory: Any,
    ) -> None:
        """Test completion respects max_tokens parameter."""
        request = chat_request_factory.simple(
            message="Write a long story about a dragon.",
            max_tokens=10,  # Very short limit
        )

        response = await client.post("/v1/chat/completions", json=request)

        assert response.status_code == 200
        data = response.json()

        # Completion tokens should be <= max_tokens
        usage = data.get("usage", {})
        assert usage.get("completion_tokens", 0) <= 15  # Allow small buffer


# =============================================================================
# Error Handling Tests
# =============================================================================


class TestErrorHandling:
    """Test error handling in e2e scenarios."""

    async def test_invalid_model_returns_404(
        self,
        client: httpx.AsyncClient,
        skip_if_service_unavailable: None,
    ) -> None:
        """Test request with invalid model returns 404."""
        request = {
            "model": "nonexistent-model-xyz",
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 10,
        }

        response = await client.post("/v1/chat/completions", json=request)

        # Should return 404 or 400 for invalid model
        assert response.status_code in [400, 404]

    async def test_invalid_request_body_returns_422(
        self,
        client: httpx.AsyncClient,
        skip_if_service_unavailable: None,
    ) -> None:
        """Test invalid request body returns 422."""
        request: dict[str, Any] = {
            "model": "phi-4",
            # Missing required 'messages' field
        }

        response = await client.post("/v1/chat/completions", json=request)

        assert response.status_code == 422

    async def test_empty_messages_returns_error(
        self,
        client: httpx.AsyncClient,
        skip_if_service_unavailable: None,
    ) -> None:
        """Test empty messages array returns error."""
        request: dict[str, Any] = {
            "model": "phi-4",
            "messages": [],
        }

        response = await client.post("/v1/chat/completions", json=request)

        # Should return validation error
        assert response.status_code in [400, 422]


# =============================================================================
# Response Format Tests
# =============================================================================


class TestResponseFormat:
    """Test response format compliance."""

    @pytest.mark.requires_model
    async def test_response_has_openai_compatible_structure(
        self,
        client: httpx.AsyncClient,
        skip_if_no_models: None,
        chat_request_factory: Any,
    ) -> None:
        """Test response follows OpenAI-compatible structure."""
        request = chat_request_factory.simple()

        response = await client.post("/v1/chat/completions", json=request)

        assert response.status_code == 200
        data = response.json()

        # OpenAI-compatible required fields
        assert "id" in data
        assert "object" in data
        assert data["object"] == "chat.completion"
        assert "created" in data
        assert "model" in data
        assert "choices" in data

    @pytest.mark.requires_model
    async def test_choice_has_required_fields(
        self,
        client: httpx.AsyncClient,
        skip_if_no_models: None,
        chat_request_factory: Any,
    ) -> None:
        """Test choice object has required fields."""
        request = chat_request_factory.simple()

        response = await client.post("/v1/chat/completions", json=request)

        assert response.status_code == 200
        data = response.json()

        choice = data["choices"][0]
        assert "index" in choice
        assert choice["index"] == 0
        assert "message" in choice
        assert "finish_reason" in choice
        assert choice["finish_reason"] in ["stop", "length", "content_filter"]

    @pytest.mark.requires_model
    async def test_message_has_role_and_content(
        self,
        client: httpx.AsyncClient,
        skip_if_no_models: None,
        chat_request_factory: Any,
    ) -> None:
        """Test message has role and content fields."""
        request = chat_request_factory.simple()

        response = await client.post("/v1/chat/completions", json=request)

        assert response.status_code == 200
        data = response.json()

        message = data["choices"][0]["message"]
        assert message["role"] == "assistant"
        assert "content" in message
        assert isinstance(message["content"], str)
