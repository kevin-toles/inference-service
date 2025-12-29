"""Orchestration mode integration tests for inference-service.

Tests multi-model orchestration modes like critique, debate, and pipeline.

AC-21.3: Multi-model test: critique mode with 2 models
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest


if TYPE_CHECKING:
    import httpx


pytestmark = [pytest.mark.integration, pytest.mark.asyncio]


# =============================================================================
# Orchestration Mode Tests
# =============================================================================


class TestOrchestrationModes:
    """Test different orchestration modes."""

    @pytest.mark.requires_model
    async def test_single_mode_uses_one_model(
        self,
        client: httpx.AsyncClient,
        skip_if_no_models: None,
        response_validator: Any,
    ) -> None:
        """Test single mode uses only one model."""
        request: dict[str, Any] = {
            "model": "llama-3.2-3b",
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 50,
            "orchestration_mode": "single",
        }

        response = await client.post("/v1/chat/completions", json=request)

        assert response.status_code == 200
        data = response.json()

        response_validator.validate_completion(data)

        # Check orchestration metadata
        if "orchestration" in data:
            orch = data["orchestration"]
            assert orch.get("mode") == "single"
            models_used = orch.get("models_used", [])
            assert len(models_used) == 1

    @pytest.mark.slow
    @pytest.mark.requires_model
    async def test_critique_mode_uses_two_models(
        self,
        client: httpx.AsyncClient,
        skip_if_no_models: None,
        response_validator: Any,
    ) -> None:
        """Test critique mode uses generator and critic models.

        AC-21.3: Critique mode uses both phi-4 and deepseek-r1-7b
        (or available models)
        """
        request: dict[str, Any] = {
            "model": "phi-4",
            "messages": [
                {"role": "user", "content": "Explain recursion in programming."}
            ],
            "max_tokens": 200,
            "orchestration_mode": "critique",
        }

        response = await client.post("/v1/chat/completions", json=request)

        # May not have enough models for critique
        if response.status_code == 400:
            pytest.skip("Not enough models loaded for critique mode")

        assert response.status_code == 200
        data = response.json()

        response_validator.validate_completion(data)

        # Validate orchestration metadata
        if "orchestration" in data:
            orch = data["orchestration"]
            assert orch.get("mode") == "critique"
            models_used = orch.get("models_used", [])
            assert len(models_used) >= 2, "Critique mode should use 2+ models"

    @pytest.mark.slow
    @pytest.mark.requires_model
    async def test_debate_mode_parallel_generation(
        self,
        client: httpx.AsyncClient,
        skip_if_no_models: None,
        response_validator: Any,
    ) -> None:
        """Test debate mode generates from multiple models in parallel."""
        request: dict[str, Any] = {
            "model": "phi-4",
            "messages": [
                {"role": "user", "content": "What is the best programming language?"}
            ],
            "max_tokens": 200,
            "orchestration_mode": "debate",
        }

        response = await client.post("/v1/chat/completions", json=request)

        if response.status_code == 400:
            pytest.skip("Not enough models loaded for debate mode")

        assert response.status_code == 200
        data = response.json()

        response_validator.validate_completion(data)

        if "orchestration" in data:
            orch = data["orchestration"]
            assert orch.get("mode") == "debate"
            # Debate should report confidence (agreement)
            if "confidence" in orch:
                assert 0.0 <= orch["confidence"] <= 1.0

    @pytest.mark.slow
    @pytest.mark.requires_model
    async def test_pipeline_mode_sequential_processing(
        self,
        client: httpx.AsyncClient,
        skip_if_no_models: None,
        response_validator: Any,
    ) -> None:
        """Test pipeline mode processes through multiple stages."""
        request: dict[str, Any] = {
            "model": "phi-4",
            "messages": [
                {"role": "user", "content": "Write a function to sort a list."}
            ],
            "max_tokens": 300,
            "orchestration_mode": "pipeline",
        }

        response = await client.post("/v1/chat/completions", json=request)

        if response.status_code == 400:
            pytest.skip("Not enough models loaded for pipeline mode")

        assert response.status_code == 200
        data = response.json()

        response_validator.validate_completion(data)

        if "orchestration" in data:
            orch = data["orchestration"]
            assert orch.get("mode") == "pipeline"


# =============================================================================
# Orchestration Metadata Tests
# =============================================================================


class TestOrchestrationMetadata:
    """Test orchestration metadata in responses."""

    @pytest.mark.requires_model
    async def test_orchestration_includes_mode(
        self,
        client: httpx.AsyncClient,
        skip_if_no_models: None,
    ) -> None:
        """Test response includes orchestration mode."""
        request: dict[str, Any] = {
            "model": "llama-3.2-3b",
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 30,
        }

        response = await client.post("/v1/chat/completions", json=request)

        assert response.status_code == 200
        data = response.json()

        if "orchestration" in data:
            assert "mode" in data["orchestration"]

    @pytest.mark.requires_model
    async def test_orchestration_includes_models_used(
        self,
        client: httpx.AsyncClient,
        skip_if_no_models: None,
    ) -> None:
        """Test response includes list of models used."""
        request: dict[str, Any] = {
            "model": "llama-3.2-3b",
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 30,
        }

        response = await client.post("/v1/chat/completions", json=request)

        assert response.status_code == 200
        data = response.json()

        if "orchestration" in data:
            models_used = data["orchestration"].get("models_used", [])
            assert isinstance(models_used, list)
            assert len(models_used) >= 1

    @pytest.mark.slow
    @pytest.mark.requires_model
    async def test_critique_reports_rounds(
        self,
        client: httpx.AsyncClient,
        skip_if_no_models: None,
    ) -> None:
        """Test critique mode reports number of rounds."""
        request: dict[str, Any] = {
            "model": "phi-4",
            "messages": [{"role": "user", "content": "Explain Python."}],
            "max_tokens": 200,
            "orchestration_mode": "critique",
        }

        response = await client.post("/v1/chat/completions", json=request)

        if response.status_code == 400:
            pytest.skip("Not enough models for critique mode")

        assert response.status_code == 200
        data = response.json()

        if "orchestration" in data:
            orch = data["orchestration"]
            if "rounds" in orch:
                assert orch["rounds"] >= 1


# =============================================================================
# Task Type Tests
# =============================================================================


class TestTaskTypes:
    """Test different task types."""

    @pytest.mark.requires_model
    async def test_code_task_type(
        self,
        client: httpx.AsyncClient,
        skip_if_no_models: None,
        chat_request_factory: Any,
    ) -> None:
        """Test code task type request."""
        request = chat_request_factory.code_task(
            message="Write a hello world function in Python."
        )

        response = await client.post("/v1/chat/completions", json=request)

        assert response.status_code == 200
        data = response.json()

        # Should get a response with code
        content = data["choices"][0]["message"]["content"]
        assert len(content) > 0

    @pytest.mark.requires_model
    async def test_reasoning_task_type(
        self,
        client: httpx.AsyncClient,
        skip_if_no_models: None,
    ) -> None:
        """Test reasoning task type request."""
        request: dict[str, Any] = {
            "model": "llama-3.2-3b",
            "messages": [
                {"role": "user", "content": "If A > B and B > C, is A > C?"}
            ],
            "max_tokens": 100,
            "task_type": "reasoning",
        }

        response = await client.post("/v1/chat/completions", json=request)

        assert response.status_code == 200


# =============================================================================
# Priority Tests
# =============================================================================


class TestRequestPriority:
    """Test request priority handling."""

    @pytest.mark.requires_model
    async def test_priority_parameter_accepted(
        self,
        client: httpx.AsyncClient,
        skip_if_no_models: None,
    ) -> None:
        """Test priority parameter is accepted."""
        request: dict[str, Any] = {
            "model": "llama-3.2-3b",
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 30,
            "priority": 3,  # High priority
        }

        response = await client.post("/v1/chat/completions", json=request)

        assert response.status_code == 200

    @pytest.mark.requires_model
    async def test_priority_levels(
        self,
        client: httpx.AsyncClient,
        skip_if_no_models: None,
    ) -> None:
        """Test different priority levels are accepted."""
        for priority in [1, 2, 3]:
            request: dict[str, Any] = {
                "model": "llama-3.2-3b",
                "messages": [{"role": "user", "content": "Hello"}],
                "max_tokens": 20,
                "priority": priority,
            }

            response = await client.post("/v1/chat/completions", json=request)

            assert response.status_code == 200, f"Priority {priority} failed"


# =============================================================================
# Orchestration Error Handling
# =============================================================================


class TestOrchestrationErrors:
    """Test error handling in orchestration modes."""

    async def test_invalid_orchestration_mode(
        self,
        client: httpx.AsyncClient,
        skip_if_service_unavailable: None,
    ) -> None:
        """Test invalid orchestration mode returns error."""
        request: dict[str, Any] = {
            "model": "llama-3.2-3b",
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 30,
            "orchestration_mode": "invalid_mode",
        }

        response = await client.post("/v1/chat/completions", json=request)

        # Should return validation error or handle gracefully
        assert response.status_code in [200, 400, 422]

    async def test_invalid_task_type(
        self,
        client: httpx.AsyncClient,
        skip_if_service_unavailable: None,
    ) -> None:
        """Test invalid task type returns error."""
        request: dict[str, Any] = {
            "model": "llama-3.2-3b",
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 30,
            "task_type": "invalid_task_type",
        }

        response = await client.post("/v1/chat/completions", json=request)

        # Should return validation error or handle gracefully
        assert response.status_code in [200, 400, 422]
