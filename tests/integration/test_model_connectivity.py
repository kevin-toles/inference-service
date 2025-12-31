"""Model Connectivity Tests - Portable across deployments.

This module tests that all available LLMs can be loaded and respond to
basic "hello" prompts. These tests are designed to work on any deployment:
- Local Mac development
- On-premise server
- AWS/Cloud deployment

Usage:
    # Local testing (default)
    pytest tests/integration/test_model_connectivity.py -v

    # Server deployment
    INFERENCE_BASE_URL=http://10.0.0.50:8085 pytest tests/integration/test_model_connectivity.py -v

    # AWS deployment with auth
    INFERENCE_BASE_URL=https://inference.prod.example.com \
    INFERENCE_API_KEY=sk-xxx \
    pytest tests/integration/test_model_connectivity.py -v

    # Smoke test for deployment verification
    pytest tests/integration/test_model_connectivity.py -v -m connectivity

Requirements:
    - inference-service accessible at INFERENCE_BASE_URL
    - At least one model available in the deployment
"""

from __future__ import annotations

import os
from typing import Any

import httpx
import pytest

from tests.integration.conftest import (
    ALL_MODELS,
    DEFAULT_TIMEOUT,
    INFERENCE_BASE_URL,
    MODEL_LOAD_TIMEOUT,
)


# =============================================================================
# Markers
# =============================================================================

pytestmark = [
    pytest.mark.integration,
    pytest.mark.connectivity,
]


# =============================================================================
# Helper Functions
# =============================================================================


async def check_model_available(client: httpx.AsyncClient, model_id: str) -> bool:
    """Check if a model is available in this deployment."""
    try:
        response = await client.get("/v1/models")
        if response.status_code == 200:
            data = response.json()
            models = data.get("data", [])
            return any(m.get("id") == model_id for m in models)
    except httpx.ConnectError:
        pass
    return False


async def check_model_loaded(client: httpx.AsyncClient, model_id: str) -> bool:
    """Check if a model is currently loaded."""
    try:
        response = await client.get("/v1/models")
        if response.status_code == 200:
            data = response.json()
            models = data.get("data", [])
            for m in models:
                if m.get("id") == model_id and m.get("status") == "loaded":
                    return True
    except httpx.ConnectError:
        pass
    return False


async def load_model(client: httpx.AsyncClient, model_id: str) -> bool:
    """Load a model if not already loaded."""
    if await check_model_loaded(client, model_id):
        return True

    try:
        response = await client.post(
            f"/v1/models/{model_id}/load",
            timeout=MODEL_LOAD_TIMEOUT,
        )
        return response.status_code in (200, 202)
    except httpx.TimeoutException:
        return False


async def send_hello_prompt(
    client: httpx.AsyncClient,
    model_id: str,
    request_timeout: float = DEFAULT_TIMEOUT,
) -> dict[str, Any]:
    """Send a simple 'hello' prompt to verify model connectivity."""
    request_payload = {
        "model": model_id,
        "messages": [
            {"role": "user", "content": "Hello, respond briefly."}
        ],
        "max_tokens": 50,
        "stream": False,
    }

    response = await client.post(
        "/v1/chat/completions",
        json=request_payload,
        timeout=request_timeout,
    )
    response.raise_for_status()
    result: dict[str, Any] = response.json()
    return result


# =============================================================================
# Service Availability Tests
# =============================================================================


class TestServiceAvailability:
    """Test that the inference service is accessible."""

    @pytest.mark.asyncio
    async def test_health_endpoint(self, client: httpx.AsyncClient) -> None:
        """Test /health endpoint is accessible."""
        response = await client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data.get("status") == "ok"

    @pytest.mark.asyncio
    async def test_models_endpoint(self, client: httpx.AsyncClient) -> None:
        """Test /v1/models endpoint returns model list."""
        response = await client.get("/v1/models")
        assert response.status_code == 200
        data = response.json()
        assert "data" in data
        assert isinstance(data["data"], list)


# =============================================================================
# Model Connectivity Tests - Parameterized for all models
# =============================================================================


class TestModelConnectivity:
    """Test connectivity for each available model.

    These tests:
    1. Check if the model exists in this deployment
    2. Load the model if not already loaded
    3. Send a simple "hello" prompt
    4. Verify valid response with content and usage stats
    """

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "model_info",
        ALL_MODELS,
        ids=[m["id"] for m in ALL_MODELS],
    )
    async def test_model_hello(
        self,
        client: httpx.AsyncClient,
        model_info: dict[str, Any],
    ) -> None:
        """Test model responds to 'hello' prompt.

        This test is the core connectivity verification - if a model
        can respond to 'hello', it's operational on this deployment.
        """
        model_id = model_info["id"]

        # Step 1: Check if model exists in this deployment
        if not await check_model_available(client, model_id):
            pytest.skip(f"Model {model_id} not available in this deployment")

        # Step 2: Load model if needed
        loaded = await load_model(client, model_id)
        if not loaded:
            pytest.skip(f"Could not load model {model_id} (may exceed memory)")

        # Step 3: Send hello prompt
        response = await send_hello_prompt(client, model_id)

        # Step 4: Validate response structure
        assert "choices" in response, "Response missing 'choices'"
        assert len(response["choices"]) > 0, "No choices in response"

        choice = response["choices"][0]
        assert "message" in choice, "Choice missing 'message'"

        message = choice["message"]
        assert message.get("role") == "assistant", "Wrong message role"
        assert "content" in message, "Message missing 'content'"
        content = message["content"]
        assert isinstance(content, str), "Content must be string"
        assert len(content.strip()) > 0, "Content should not be empty"

        # Step 5: Validate usage stats
        assert "usage" in response, "Response missing 'usage'"
        usage = response["usage"]
        assert usage.get("total_tokens", 0) > 0, "total_tokens should be > 0"


# =============================================================================
# Individual Model Tests (for CI/CD)
# =============================================================================


class TestIndividualModels:
    """Individual model tests for granular CI/CD reporting."""

    @pytest.mark.asyncio
    async def test_deepseek_r1_7b_connectivity(
        self, client: httpx.AsyncClient
    ) -> None:
        """Test deepseek-r1-7b responds to prompts."""
        model_id = "deepseek-r1-7b"
        if not await check_model_available(client, model_id):
            pytest.skip(f"Model {model_id} not available")
        if not await load_model(client, model_id):
            pytest.skip(f"Could not load {model_id}")

        response = await send_hello_prompt(client, model_id)
        assert response["choices"][0]["message"]["content"].strip()
        assert response["usage"]["total_tokens"] > 0

    @pytest.mark.asyncio
    async def test_qwen25_7b_connectivity(self, client: httpx.AsyncClient) -> None:
        """Test qwen2.5-7b responds to prompts."""
        model_id = "qwen2.5-7b"
        if not await check_model_available(client, model_id):
            pytest.skip(f"Model {model_id} not available")
        if not await load_model(client, model_id):
            pytest.skip(f"Could not load {model_id}")

        response = await send_hello_prompt(client, model_id)
        assert response["choices"][0]["message"]["content"].strip()
        assert response["usage"]["total_tokens"] > 0

    @pytest.mark.asyncio
    async def test_phi4_connectivity(self, client: httpx.AsyncClient) -> None:
        """Test phi-4 responds to prompts."""
        model_id = "phi-4"
        if not await check_model_available(client, model_id):
            pytest.skip(f"Model {model_id} not available")
        if not await load_model(client, model_id):
            pytest.skip(f"Could not load {model_id}")

        response = await send_hello_prompt(client, model_id)
        assert response["choices"][0]["message"]["content"].strip()
        assert response["usage"]["total_tokens"] > 0

    @pytest.mark.asyncio
    async def test_llama32_3b_connectivity(self, client: httpx.AsyncClient) -> None:
        """Test llama-3.2-3b responds to prompts."""
        model_id = "llama-3.2-3b"
        if not await check_model_available(client, model_id):
            pytest.skip(f"Model {model_id} not available")
        if not await load_model(client, model_id):
            pytest.skip(f"Could not load {model_id}")

        response = await send_hello_prompt(client, model_id)
        assert response["choices"][0]["message"]["content"].strip()
        assert response["usage"]["total_tokens"] > 0

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_phi3_medium_128k_connectivity(
        self, client: httpx.AsyncClient
    ) -> None:
        """Test phi-3-medium-128k responds to prompts (slow - large model)."""
        model_id = "phi-3-medium-128k"
        if not await check_model_available(client, model_id):
            pytest.skip(f"Model {model_id} not available")
        if not await load_model(client, model_id):
            pytest.skip(f"Could not load {model_id}")

        response = await send_hello_prompt(client, model_id, timeout=180.0)
        assert response["choices"][0]["message"]["content"].strip()
        assert response["usage"]["total_tokens"] > 0

    @pytest.mark.asyncio
    async def test_granite_8b_code_128k_connectivity(
        self, client: httpx.AsyncClient
    ) -> None:
        """Test granite-8b-code-128k responds to prompts."""
        model_id = "granite-8b-code-128k"
        if not await check_model_available(client, model_id):
            pytest.skip(f"Model {model_id} not available")
        if not await load_model(client, model_id):
            pytest.skip(f"Could not load {model_id}")

        response = await send_hello_prompt(client, model_id)
        assert response["choices"][0]["message"]["content"].strip()
        assert response["usage"]["total_tokens"] > 0


# =============================================================================
# Deployment Summary Test
# =============================================================================


class TestDeploymentSummary:
    """Summary tests for deployment verification."""

    @pytest.mark.asyncio
    async def test_deployment_info(self, client: httpx.AsyncClient) -> None:
        """Print deployment info for verification."""
        print(f"\n{'='*60}")
        print("INFERENCE SERVICE DEPLOYMENT INFO")
        print(f"{'='*60}")
        print(f"Base URL: {INFERENCE_BASE_URL}")
        print(f"Timeout: {DEFAULT_TIMEOUT}s")
        print(f"API Key: {'[SET]' if os.getenv('INFERENCE_API_KEY') else '[NOT SET]'}")

        # Get health
        response = await client.get("/health")
        print(f"Health Status: {response.status_code}")

        # Get models
        response = await client.get("/v1/models")
        if response.status_code == 200:
            data = response.json()
            models = data.get("data", [])
            print(f"\nAvailable Models ({len(models)}):")
            for m in models:
                status = m.get("status", "unknown")
                size = m.get("size_gb", "?")
                print(f"  - {m.get('id')}: {status} ({size}GB)")

        print(f"{'='*60}\n")

        # This test always passes - it's just for info
        assert True
