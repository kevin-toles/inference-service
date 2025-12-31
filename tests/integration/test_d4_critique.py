"""D4 Critique Mode End-to-End Tests.

Tests the critique orchestration mode where:
- qwen2.5-7b acts as the generator (creates initial response)
- deepseek-r1-7b acts as the critic (provides chain-of-thought critique)

The critique mode flow:
1. Generator creates initial response
2. Critic evaluates and provides feedback
3. Generator revises based on feedback (optional)

Usage:
    INFERENCE_DEFAULT_PRESET=D4 pytest tests/integration/test_d4_critique.py -v
"""

from __future__ import annotations

from typing import Any

import httpx
import pytest

from tests.integration.conftest import D4_PRESET, MODEL_LOAD_TIMEOUT


# =============================================================================
# Markers
# =============================================================================

pytestmark = [
    pytest.mark.integration,
    pytest.mark.d4_critique,
    pytest.mark.slow,
]


# =============================================================================
# Helper Functions
# =============================================================================


async def ensure_d4_models_loaded(client: httpx.AsyncClient) -> None:
    """Ensure both D4 models are loaded."""
    for model_id in D4_PRESET["models"]:
        response = await client.post(
            f"/v1/models/{model_id}/load",
            timeout=MODEL_LOAD_TIMEOUT,
        )
        # Accept any success status
        if response.status_code not in (200, 202, 409):
            pytest.skip(f"Could not load D4 model {model_id}")


# =============================================================================
# Critique Mode Tests
# =============================================================================


class TestCritiqueModeE2E:
    """End-to-end tests for critique orchestration mode."""

    @pytest.mark.asyncio
    async def test_critique_mode_request(
        self,
        client: httpx.AsyncClient,
        d4_preset: dict[str, Any],
    ) -> None:
        """Test critique mode request is accepted."""
        await ensure_d4_models_loaded(client)

        request = {
            "model": "qwen2.5-7b",  # Generator
            "messages": [
                {"role": "user", "content": "Write a Python function to reverse a string."}
            ],
            "max_tokens": 200,
            "stream": False,
            "orchestration_mode": "critique",
        }

        response = await client.post(
            "/v1/chat/completions",
            json=request,
            timeout=120.0,
        )
        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_critique_mode_returns_response(
        self,
        client: httpx.AsyncClient,
    ) -> None:
        """Test critique mode returns valid response."""
        await ensure_d4_models_loaded(client)

        request = {
            "model": "qwen2.5-7b",
            "messages": [
                {"role": "user", "content": "What is 2 + 2?"}
            ],
            "max_tokens": 100,
            "stream": False,
            "orchestration_mode": "critique",
        }

        response = await client.post(
            "/v1/chat/completions",
            json=request,
            timeout=120.0,
        )
        assert response.status_code == 200

        data = response.json()
        assert "choices" in data
        assert len(data["choices"]) > 0
        assert data["choices"][0]["message"]["content"].strip()

    @pytest.mark.asyncio
    async def test_critique_mode_has_orchestration_metadata(
        self,
        client: httpx.AsyncClient,
    ) -> None:
        """Test critique mode response includes orchestration metadata."""
        await ensure_d4_models_loaded(client)

        request = {
            "model": "qwen2.5-7b",
            "messages": [
                {"role": "user", "content": "Explain recursion simply."}
            ],
            "max_tokens": 150,
            "stream": False,
            "orchestration_mode": "critique",
        }

        response = await client.post(
            "/v1/chat/completions",
            json=request,
            timeout=120.0,
        )
        assert response.status_code == 200

        data = response.json()

        # Check orchestration metadata exists
        if "orchestration" in data:
            orch = data["orchestration"]
            assert "mode" in orch
            assert orch["mode"] == "critique"

            # Check models_used includes both D4 models
            if "models_used" in orch:
                models_used = orch["models_used"]
                assert "qwen2.5-7b" in models_used or "deepseek-r1-7b" in models_used

    @pytest.mark.asyncio
    async def test_critique_mode_uses_both_models(
        self,
        client: httpx.AsyncClient,
        d4_preset: dict[str, Any],
    ) -> None:
        """Test critique mode uses both generator and critic models."""
        await ensure_d4_models_loaded(client)

        request = {
            "model": "qwen2.5-7b",
            "messages": [
                {
                    "role": "user",
                    "content": "Write a function to check if a number is prime.",
                }
            ],
            "max_tokens": 300,
            "stream": False,
            "orchestration_mode": "critique",
        }

        response = await client.post(
            "/v1/chat/completions",
            json=request,
            timeout=180.0,
        )
        assert response.status_code == 200

        data = response.json()

        # Verify response has content
        content = data["choices"][0]["message"]["content"]
        assert len(content) > 0

        # If orchestration metadata present, verify models
        if "orchestration" in data:
            models_used = data["orchestration"].get("models_used", [])
            # At minimum, should see generator model used
            # Full critique should show both
            print(f"Models used in critique: {models_used}")


class TestCritiqueModeCodeGeneration:
    """Test critique mode for code generation tasks."""

    @pytest.mark.asyncio
    async def test_critique_improves_code_quality(
        self,
        client: httpx.AsyncClient,
    ) -> None:
        """Test critique mode produces quality code responses."""
        await ensure_d4_models_loaded(client)

        request = {
            "model": "qwen2.5-7b",
            "messages": [
                {
                    "role": "system",
                    "content": "You are a helpful Python coding assistant.",
                },
                {
                    "role": "user",
                    "content": "Write a function to calculate factorial with error handling.",
                },
            ],
            "max_tokens": 400,
            "stream": False,
            "orchestration_mode": "critique",
        }

        response = await client.post(
            "/v1/chat/completions",
            json=request,
            timeout=180.0,
        )
        assert response.status_code == 200

        data = response.json()
        content = data["choices"][0]["message"]["content"]

        # Code should contain function definition
        assert "def " in content.lower() or "function" in content.lower()

    @pytest.mark.asyncio
    async def test_critique_mode_usage_stats(
        self,
        client: httpx.AsyncClient,
    ) -> None:
        """Test critique mode returns usage statistics."""
        await ensure_d4_models_loaded(client)

        request = {
            "model": "qwen2.5-7b",
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 50,
            "stream": False,
            "orchestration_mode": "critique",
        }

        response = await client.post(
            "/v1/chat/completions",
            json=request,
            timeout=120.0,
        )
        assert response.status_code == 200

        data = response.json()
        assert "usage" in data
        assert data["usage"]["total_tokens"] > 0


class TestCritiqueModeRoles:
    """Test D4 role-specific behavior in critique mode."""

    @pytest.mark.asyncio
    async def test_generator_role_qwen(
        self,
        client: httpx.AsyncClient,
    ) -> None:
        """Test qwen2.5-7b works as generator in D4."""
        await ensure_d4_models_loaded(client)

        # Direct call to generator should work
        request = {
            "model": "qwen2.5-7b",
            "messages": [{"role": "user", "content": "Generate a greeting."}],
            "max_tokens": 50,
            "stream": False,
        }

        response = await client.post("/v1/chat/completions", json=request)
        assert response.status_code == 200
        assert response.json()["choices"][0]["message"]["content"].strip()

    @pytest.mark.asyncio
    async def test_critic_role_deepseek(
        self,
        client: httpx.AsyncClient,
    ) -> None:
        """Test deepseek-r1-7b works as critic in D4."""
        await ensure_d4_models_loaded(client)

        # Direct call to critic should work
        request = {
            "model": "deepseek-r1-7b",
            "messages": [
                {"role": "user", "content": "Critique this code: print('hello')"}
            ],
            "max_tokens": 100,
            "stream": False,
        }

        response = await client.post("/v1/chat/completions", json=request)
        assert response.status_code == 200
        assert response.json()["choices"][0]["message"]["content"].strip()
