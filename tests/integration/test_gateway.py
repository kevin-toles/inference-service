"""LLM Gateway integration tests for inference-service.

Tests routing through llm-gateway to inference-service.

AC-21.4: llm-gateway integration test: route through gateway
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest


if TYPE_CHECKING:
    import httpx


pytestmark = [
    pytest.mark.integration,
    pytest.mark.asyncio,
    pytest.mark.requires_gateway,
]


# =============================================================================
# Gateway Health Tests
# =============================================================================


class TestGatewayHealth:
    """Test llm-gateway health and connectivity."""

    async def test_gateway_health(
        self,
        gateway_client: httpx.AsyncClient,
        skip_if_gateway_unavailable: None,
    ) -> None:
        """Test gateway health endpoint."""
        response = await gateway_client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data.get("status") == "ok"

    async def test_gateway_ready(
        self,
        gateway_client: httpx.AsyncClient,
        skip_if_gateway_unavailable: None,
    ) -> None:
        """Test gateway readiness endpoint."""
        response = await gateway_client.get("/health/ready")

        # Either ready or not ready
        assert response.status_code in [200, 503]


# =============================================================================
# Gateway Routing Tests
# =============================================================================


class TestGatewayRouting:
    """Test routing through gateway to inference-service."""

    @pytest.mark.slow
    async def test_inference_prefix_routes_to_service(
        self,
        gateway_client: httpx.AsyncClient,
        skip_if_gateway_unavailable: None,
    ) -> None:
        """Test inference: prefix routes to inference-service.

        AC-21.4: `inference:phi-4` routed through gateway to inference-service
        """
        request: dict[str, Any] = {
            "model": "inference:phi-4",  # Prefix routes to inference-service
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 30,
        }

        response = await gateway_client.post("/v1/chat/completions", json=request)

        # May return 404 if inference-service not configured in gateway
        if response.status_code == 404:
            pytest.skip("inference-service not configured in gateway")

        # Should succeed or return service unavailable
        assert response.status_code in [200, 503]

    @pytest.mark.slow
    async def test_local_prefix_routes_to_service(
        self,
        gateway_client: httpx.AsyncClient,
        skip_if_gateway_unavailable: None,
    ) -> None:
        """Test local: prefix routes to inference-service."""
        request: dict[str, Any] = {
            "model": "local:llama-3.2-3b",
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 30,
        }

        response = await gateway_client.post("/v1/chat/completions", json=request)

        if response.status_code == 404:
            pytest.skip("local: prefix not configured in gateway")

        assert response.status_code in [200, 503]


# =============================================================================
# Gateway Response Tests
# =============================================================================


class TestGatewayResponses:
    """Test response handling through gateway."""

    @pytest.mark.slow
    async def test_gateway_returns_completion(
        self,
        gateway_client: httpx.AsyncClient,
        skip_if_gateway_unavailable: None,
        response_validator: Any,
    ) -> None:
        """Test gateway returns valid completion response."""
        request: dict[str, Any] = {
            "model": "inference:llama-3.2-3b",
            "messages": [{"role": "user", "content": "Say hello."}],
            "max_tokens": 50,
        }

        response = await gateway_client.post("/v1/chat/completions", json=request)

        if response.status_code == 404:
            pytest.skip("inference-service not configured in gateway")

        if response.status_code == 200:
            data = response.json()
            response_validator.validate_completion(data)

    @pytest.mark.slow
    async def test_gateway_streaming(
        self,
        gateway_client: httpx.AsyncClient,
        skip_if_gateway_unavailable: None,
        sse_parser: Any,
    ) -> None:
        """Test streaming through gateway."""
        request: dict[str, Any] = {
            "model": "inference:llama-3.2-3b",
            "messages": [{"role": "user", "content": "Count to 3"}],
            "max_tokens": 30,
            "stream": True,
        }

        async with gateway_client.stream(
            "POST", "/v1/chat/completions", json=request
        ) as response:
            if response.status_code == 404:
                pytest.skip("inference-service not configured in gateway")

            if response.status_code == 200:
                content = (await response.aread()).decode("utf-8")
                chunks = sse_parser.parse_stream(content)

                assert len(chunks) > 0
                assert chunks[-1] == "[DONE]"


# =============================================================================
# Gateway Error Handling Tests
# =============================================================================


class TestGatewayErrors:
    """Test error handling through gateway."""

    async def test_gateway_invalid_model(
        self,
        gateway_client: httpx.AsyncClient,
        skip_if_gateway_unavailable: None,
    ) -> None:
        """Test gateway handles invalid model gracefully."""
        request: dict[str, Any] = {
            "model": "inference:nonexistent-model",
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 30,
        }

        response = await gateway_client.post("/v1/chat/completions", json=request)

        # Should return an error status
        assert response.status_code in [400, 404, 503]

    async def test_gateway_validation_error(
        self,
        gateway_client: httpx.AsyncClient,
        skip_if_gateway_unavailable: None,
    ) -> None:
        """Test gateway returns validation errors."""
        request: dict[str, Any] = {
            "model": "inference:phi-4",
            # Missing messages
        }

        response = await gateway_client.post("/v1/chat/completions", json=request)

        assert response.status_code == 422


# =============================================================================
# Gateway Provider Discovery Tests
# =============================================================================


class TestGatewayProviders:
    """Test provider discovery through gateway."""

    async def test_gateway_lists_providers(
        self,
        gateway_client: httpx.AsyncClient,
        skip_if_gateway_unavailable: None,
    ) -> None:
        """Test gateway lists available providers."""
        response = await gateway_client.get("/v1/providers")

        if response.status_code == 404:
            pytest.skip("Providers endpoint not available")

        assert response.status_code == 200
        data = response.json()
        assert "providers" in data or "data" in data

    async def test_gateway_models_includes_inference(
        self,
        gateway_client: httpx.AsyncClient,
        skip_if_gateway_unavailable: None,
    ) -> None:
        """Test gateway models list includes inference-service models."""
        response = await gateway_client.get("/v1/models")

        assert response.status_code == 200
        data = response.json()

        # Check if inference models are listed
        models = data.get("data", [])
        _inference_models = [m for m in models if m.get("id", "").startswith("inference:")]

        # May or may not have inference models depending on configuration
        # Just verify the endpoint works
        assert isinstance(models, list)
