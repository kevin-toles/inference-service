"""D4 Preset Load Tests - Verify D4 configuration loads correctly.

D4 Preset:
- Models: deepseek-r1-7b (4.7GB) + qwen2.5-7b (4.5GB) = 9.2GB total
- Mode: critique (generator + critic)
- Roles: qwen2.5-7b=generator, deepseek-r1-7b=critic

Usage:
    INFERENCE_DEFAULT_PRESET=D4 pytest tests/integration/test_d4_preset.py -v

    # Remote deployment
    INFERENCE_BASE_URL=http://10.0.0.50:8085 \
    INFERENCE_DEFAULT_PRESET=D4 \
    pytest tests/integration/test_d4_preset.py -v
"""

from __future__ import annotations

from typing import Any

import httpx
import pytest

from tests.integration.conftest import MODEL_LOAD_TIMEOUT


# =============================================================================
# Markers
# =============================================================================

pytestmark = [
    pytest.mark.integration,
    pytest.mark.d4_preset,
]


# =============================================================================
# D4 Preset Tests
# =============================================================================


class TestD4PresetLoad:
    """Test that D4 preset loads correctly."""

    @pytest.mark.asyncio
    async def test_d4_models_available(
        self,
        client: httpx.AsyncClient,
        d4_preset: dict[str, Any],
    ) -> None:
        """Verify D4 models are available in the deployment."""
        response = await client.get("/v1/models")
        assert response.status_code == 200

        data = response.json()
        available_ids = [m.get("id") for m in data.get("data", [])]

        for model_id in d4_preset["models"]:
            assert model_id in available_ids, (
                f"D4 model {model_id} not available. "
                f"Available: {available_ids}"
            )

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_d4_models_can_load(
        self,
        client: httpx.AsyncClient,
        d4_preset: dict[str, Any],
    ) -> None:
        """Verify D4 models can be loaded (9.2GB total)."""
        for model_id in d4_preset["models"]:
            response = await client.post(
                f"/v1/models/{model_id}/load",
                timeout=MODEL_LOAD_TIMEOUT,
            )
            # Accept 200 (loaded), 202 (loading), or 409 (already loaded)
            assert response.status_code in (200, 202, 409), (
                f"Failed to load {model_id}: {response.status_code}"
            )

    @pytest.mark.asyncio
    async def test_d4_models_loaded_status(
        self,
        client: httpx.AsyncClient,
        d4_preset: dict[str, Any],
    ) -> None:
        """Verify D4 models show as loaded after loading."""
        # First ensure models are loaded
        for model_id in d4_preset["models"]:
            await client.post(
                f"/v1/models/{model_id}/load",
                timeout=MODEL_LOAD_TIMEOUT,
            )

        # Check status
        response = await client.get("/v1/models")
        assert response.status_code == 200

        data = response.json()
        model_statuses = {
            m.get("id"): m.get("status")
            for m in data.get("data", [])
        }

        for model_id in d4_preset["models"]:
            assert model_statuses.get(model_id) == "loaded", (
                f"D4 model {model_id} not loaded. Status: {model_statuses.get(model_id)}"
            )

    @pytest.mark.asyncio
    async def test_d4_memory_budget(
        self,
        client: httpx.AsyncClient,
        d4_preset: dict[str, Any],
    ) -> None:
        """Verify D4 models fit within memory budget."""
        response = await client.get("/v1/models")
        assert response.status_code == 200

        data = response.json()
        total_loaded_size = 0.0

        for m in data.get("data", []):
            if m.get("id") in d4_preset["models"] and m.get("status") == "loaded":
                total_loaded_size += m.get("size_gb", 0)

        # D4 should be ~9.2GB
        assert total_loaded_size <= 16.0, (
            f"D4 models exceed 16GB budget: {total_loaded_size}GB"
        )

    @pytest.mark.asyncio
    async def test_d4_generator_responds(
        self,
        client: httpx.AsyncClient,
        d4_preset: dict[str, Any],
    ) -> None:
        """Test D4 generator model (qwen2.5-7b) responds."""
        generator_model = "qwen2.5-7b"

        request = {
            "model": generator_model,
            "messages": [{"role": "user", "content": "Hello, respond briefly."}],
            "max_tokens": 50,
            "stream": False,
        }

        response = await client.post("/v1/chat/completions", json=request)
        assert response.status_code == 200

        data = response.json()
        content = data["choices"][0]["message"]["content"]
        assert len(content.strip()) > 0

    @pytest.mark.asyncio
    async def test_d4_critic_responds(
        self,
        client: httpx.AsyncClient,
        d4_preset: dict[str, Any],
    ) -> None:
        """Test D4 critic model (deepseek-r1-7b) responds."""
        critic_model = "deepseek-r1-7b"

        request = {
            "model": critic_model,
            "messages": [{"role": "user", "content": "Hello, respond briefly."}],
            "max_tokens": 50,
            "stream": False,
        }

        response = await client.post("/v1/chat/completions", json=request)
        assert response.status_code == 200

        data = response.json()
        content = data["choices"][0]["message"]["content"]
        assert len(content.strip()) > 0


class TestD4PresetRoles:
    """Test D4 role assignments work correctly."""

    @pytest.mark.asyncio
    async def test_d4_role_config(self, d4_preset: dict[str, Any]) -> None:
        """Verify D4 role configuration is correct."""
        roles = d4_preset["roles"]

        assert roles["qwen2.5-7b"] == "generator"
        assert roles["deepseek-r1-7b"] == "critic"

    @pytest.mark.asyncio
    async def test_d4_orchestration_mode(self, d4_preset: dict[str, Any]) -> None:
        """Verify D4 uses critique orchestration mode."""
        assert d4_preset["orchestration_mode"] == "critique"
