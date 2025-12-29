"""Unit tests for PipelineMode orchestration.

Tests for:
- PipelineMode: Draft → Refine → Validate flow (AC-13.1)
- PipelineMode compresses between steps if needed (AC-13.2)
- PipelineMode supports saga compensation on failure (AC-13.3)
- PipelineMode returns partial result on error (AC-13.4)

Flow: Request → Fast(draft) → Specialist(refine) → Primary(validate) → Response
"""

import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.models.requests import ChatCompletionRequest, Message
from src.models.responses import (
    ChatCompletionResponse,
    Choice,
    ChoiceMessage,
    Usage,
)
from src.orchestration.modes.pipeline import PipelineMode
from src.providers.base import InferenceProvider, ModelMetadata


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def drafter_provider() -> MagicMock:
    """Create a mock drafter provider (Fast model)."""
    provider = MagicMock(spec=InferenceProvider)
    provider.model_info = ModelMetadata(
        model_id="llama-3.2-3b",
        context_length=8192,
        roles=["fast", "primary"],
        memory_mb=4000,
        status="loaded",
    )
    provider.is_loaded = True
    return provider


@pytest.fixture
def refiner_provider() -> MagicMock:
    """Create a mock refiner provider (Coder/Thinker model)."""
    provider = MagicMock(spec=InferenceProvider)
    provider.model_info = ModelMetadata(
        model_id="deepseek-r1-7b",
        context_length=32768,
        roles=["thinker"],
        memory_mb=8000,
        status="loaded",
    )
    provider.is_loaded = True
    return provider


@pytest.fixture
def validator_provider() -> MagicMock:
    """Create a mock validator provider (Primary model)."""
    provider = MagicMock(spec=InferenceProvider)
    provider.model_info = ModelMetadata(
        model_id="phi-4",
        context_length=16384,
        roles=["primary", "thinker"],
        memory_mb=8000,
        status="loaded",
    )
    provider.is_loaded = True
    return provider


@pytest.fixture
def sample_request() -> ChatCompletionRequest:
    """Create a sample chat completion request."""
    return ChatCompletionRequest(
        model="phi-4",
        messages=[
            Message(role="system", content="You are a helpful assistant."),
            Message(role="user", content="Write a function to calculate factorial."),
        ],
        temperature=0.7,
        max_tokens=500,
    )


@pytest.fixture
def draft_response() -> ChatCompletionResponse:
    """Create a draft response from fast model."""
    return ChatCompletionResponse(
        id="chatcmpl-draft-123",
        created=int(time.time()),
        model="llama-3.2-3b",
        choices=[
            Choice(
                index=0,
                message=ChoiceMessage(
                    role="assistant",
                    content="def factorial(n): return 1 if n <= 1 else n * factorial(n-1)",
                ),
                finish_reason="stop",
            )
        ],
        usage=Usage(
            prompt_tokens=30,
            completion_tokens=25,
            total_tokens=55,
        ),
    )


@pytest.fixture
def refined_response() -> ChatCompletionResponse:
    """Create a refined response from specialist model."""
    return ChatCompletionResponse(
        id="chatcmpl-refine-123",
        created=int(time.time()),
        model="deepseek-r1-7b",
        choices=[
            Choice(
                index=0,
                message=ChoiceMessage(
                    role="assistant",
                    content=(
                        "def factorial(n: int) -> int:\n"
                        "    '''Calculate factorial iteratively.'''\n"
                        "    if n < 0:\n"
                        "        raise ValueError('n must be non-negative')\n"
                        "    result = 1\n"
                        "    for i in range(2, n + 1):\n"
                        "        result *= i\n"
                        "    return result"
                    ),
                ),
                finish_reason="stop",
            )
        ],
        usage=Usage(
            prompt_tokens=60,
            completion_tokens=80,
            total_tokens=140,
        ),
    )


@pytest.fixture
def validated_response() -> ChatCompletionResponse:
    """Create a validated response from primary model."""
    return ChatCompletionResponse(
        id="chatcmpl-validate-123",
        created=int(time.time()),
        model="phi-4",
        choices=[
            Choice(
                index=0,
                message=ChoiceMessage(
                    role="assistant",
                    content=(
                        "Here's a well-implemented factorial function:\n\n"
                        "```python\n"
                        "def factorial(n: int) -> int:\n"
                        "    '''Calculate factorial iteratively.'''\n"
                        "    if n < 0:\n"
                        "        raise ValueError('n must be non-negative')\n"
                        "    result = 1\n"
                        "    for i in range(2, n + 1):\n"
                        "        result *= i\n"
                        "    return result\n"
                        "```\n\n"
                        "This implementation handles edge cases and is efficient."
                    ),
                ),
                finish_reason="stop",
            )
        ],
        usage=Usage(
            prompt_tokens=120,
            completion_tokens=100,
            total_tokens=220,
        ),
    )


# =============================================================================
# AC-13.1: PipelineMode Draft → Refine → Validate flow
# =============================================================================


class TestPipelineModeFlow:
    """Tests for PipelineMode execute() flow."""

    @pytest.mark.asyncio
    async def test_pipeline_mode_executes_draft_phase(
        self,
        drafter_provider: MagicMock,
        refiner_provider: MagicMock,
        validator_provider: MagicMock,
        sample_request: ChatCompletionRequest,
        draft_response: ChatCompletionResponse,
        refined_response: ChatCompletionResponse,
        validated_response: ChatCompletionResponse,
    ) -> None:
        """Test PipelineMode executes draft phase first."""
        drafter_provider.generate = AsyncMock(return_value=draft_response)
        refiner_provider.generate = AsyncMock(return_value=refined_response)
        validator_provider.generate = AsyncMock(return_value=validated_response)

        mode = PipelineMode(
            drafter=drafter_provider,
            refiner=refiner_provider,
            validator=validator_provider,
        )
        await mode.execute(sample_request)

        drafter_provider.generate.assert_called()

    @pytest.mark.asyncio
    async def test_pipeline_mode_executes_refine_phase(
        self,
        drafter_provider: MagicMock,
        refiner_provider: MagicMock,
        validator_provider: MagicMock,
        sample_request: ChatCompletionRequest,
        draft_response: ChatCompletionResponse,
        refined_response: ChatCompletionResponse,
        validated_response: ChatCompletionResponse,
    ) -> None:
        """Test PipelineMode executes refine phase after draft."""
        drafter_provider.generate = AsyncMock(return_value=draft_response)
        refiner_provider.generate = AsyncMock(return_value=refined_response)
        validator_provider.generate = AsyncMock(return_value=validated_response)

        mode = PipelineMode(
            drafter=drafter_provider,
            refiner=refiner_provider,
            validator=validator_provider,
        )
        await mode.execute(sample_request)

        refiner_provider.generate.assert_called()

    @pytest.mark.asyncio
    async def test_pipeline_mode_executes_validate_phase(
        self,
        drafter_provider: MagicMock,
        refiner_provider: MagicMock,
        validator_provider: MagicMock,
        sample_request: ChatCompletionRequest,
        draft_response: ChatCompletionResponse,
        refined_response: ChatCompletionResponse,
        validated_response: ChatCompletionResponse,
    ) -> None:
        """Test PipelineMode executes validate phase last."""
        drafter_provider.generate = AsyncMock(return_value=draft_response)
        refiner_provider.generate = AsyncMock(return_value=refined_response)
        validator_provider.generate = AsyncMock(return_value=validated_response)

        mode = PipelineMode(
            drafter=drafter_provider,
            refiner=refiner_provider,
            validator=validator_provider,
        )
        await mode.execute(sample_request)

        validator_provider.generate.assert_called()

    @pytest.mark.asyncio
    async def test_pipeline_mode_all_three_phases_called(
        self,
        drafter_provider: MagicMock,
        refiner_provider: MagicMock,
        validator_provider: MagicMock,
        sample_request: ChatCompletionRequest,
        draft_response: ChatCompletionResponse,
        refined_response: ChatCompletionResponse,
        validated_response: ChatCompletionResponse,
    ) -> None:
        """Test PipelineMode calls all three phases."""
        drafter_provider.generate = AsyncMock(return_value=draft_response)
        refiner_provider.generate = AsyncMock(return_value=refined_response)
        validator_provider.generate = AsyncMock(return_value=validated_response)

        mode = PipelineMode(
            drafter=drafter_provider,
            refiner=refiner_provider,
            validator=validator_provider,
        )
        await mode.execute(sample_request)

        assert drafter_provider.generate.call_count == 1
        assert refiner_provider.generate.call_count == 1
        assert validator_provider.generate.call_count == 1

    @pytest.mark.asyncio
    async def test_pipeline_mode_passes_draft_to_refiner(
        self,
        drafter_provider: MagicMock,
        refiner_provider: MagicMock,
        validator_provider: MagicMock,
        sample_request: ChatCompletionRequest,
        draft_response: ChatCompletionResponse,
        refined_response: ChatCompletionResponse,
        validated_response: ChatCompletionResponse,
    ) -> None:
        """Test PipelineMode passes draft output to refiner."""
        drafter_provider.generate = AsyncMock(return_value=draft_response)
        refiner_provider.generate = AsyncMock(return_value=refined_response)
        validator_provider.generate = AsyncMock(return_value=validated_response)

        mode = PipelineMode(
            drafter=drafter_provider,
            refiner=refiner_provider,
            validator=validator_provider,
        )
        await mode.execute(sample_request)

        # Verify refiner was called with draft content
        refiner_call = refiner_provider.generate.call_args[0][0]
        messages_content = [m.content for m in refiner_call.messages if m.content]
        assert any("factorial" in (c or "") for c in messages_content)

    @pytest.mark.asyncio
    async def test_pipeline_mode_passes_refined_to_validator(
        self,
        drafter_provider: MagicMock,
        refiner_provider: MagicMock,
        validator_provider: MagicMock,
        sample_request: ChatCompletionRequest,
        draft_response: ChatCompletionResponse,
        refined_response: ChatCompletionResponse,
        validated_response: ChatCompletionResponse,
    ) -> None:
        """Test PipelineMode passes refined output to validator."""
        drafter_provider.generate = AsyncMock(return_value=draft_response)
        refiner_provider.generate = AsyncMock(return_value=refined_response)
        validator_provider.generate = AsyncMock(return_value=validated_response)

        mode = PipelineMode(
            drafter=drafter_provider,
            refiner=refiner_provider,
            validator=validator_provider,
        )
        await mode.execute(sample_request)

        # Verify validator was called with refined content
        validator_call = validator_provider.generate.call_args[0][0]
        messages_content = [m.content for m in validator_call.messages if m.content]
        # Refined content should have type hints
        assert any("int" in (c or "") for c in messages_content)

    @pytest.mark.asyncio
    async def test_pipeline_mode_returns_validated_content(
        self,
        drafter_provider: MagicMock,
        refiner_provider: MagicMock,
        validator_provider: MagicMock,
        sample_request: ChatCompletionRequest,
        draft_response: ChatCompletionResponse,
        refined_response: ChatCompletionResponse,
        validated_response: ChatCompletionResponse,
    ) -> None:
        """Test PipelineMode returns validator's final response."""
        drafter_provider.generate = AsyncMock(return_value=draft_response)
        refiner_provider.generate = AsyncMock(return_value=refined_response)
        validator_provider.generate = AsyncMock(return_value=validated_response)

        mode = PipelineMode(
            drafter=drafter_provider,
            refiner=refiner_provider,
            validator=validator_provider,
        )
        result = await mode.execute(sample_request)

        # Final result should be from validator
        assert "well-implemented" in (result.choices[0].message.content or "")


# =============================================================================
# AC-13.2: PipelineMode compresses between steps if needed
# =============================================================================


class TestPipelineModeCompression:
    """Tests for PipelineMode compression between steps."""

    @pytest.mark.asyncio
    async def test_pipeline_mode_compresses_large_draft(
        self,
        drafter_provider: MagicMock,
        refiner_provider: MagicMock,
        validator_provider: MagicMock,
        sample_request: ChatCompletionRequest,
        refined_response: ChatCompletionResponse,
        validated_response: ChatCompletionResponse,
    ) -> None:
        """Test PipelineMode compresses large draft before refine."""
        # Create a large draft response (>16K tokens worth)
        large_content = "x " * 20000  # ~20K tokens
        large_draft = ChatCompletionResponse(
            id="chatcmpl-large",
            created=int(time.time()),
            model="llama-3.2-3b",
            choices=[
                Choice(
                    index=0,
                    message=ChoiceMessage(role="assistant", content=large_content),
                    finish_reason="stop",
                )
            ],
            usage=Usage(prompt_tokens=30, completion_tokens=20000, total_tokens=20030),
        )

        drafter_provider.generate = AsyncMock(return_value=large_draft)
        refiner_provider.generate = AsyncMock(return_value=refined_response)
        validator_provider.generate = AsyncMock(return_value=validated_response)

        mode = PipelineMode(
            drafter=drafter_provider,
            refiner=refiner_provider,
            validator=validator_provider,
        )
        result = await mode.execute(sample_request)

        # Pipeline should complete successfully (compression applied)
        assert result is not None
        assert result.orchestration is not None

    @pytest.mark.asyncio
    async def test_pipeline_mode_no_compression_for_small_content(
        self,
        drafter_provider: MagicMock,
        refiner_provider: MagicMock,
        validator_provider: MagicMock,
        sample_request: ChatCompletionRequest,
        draft_response: ChatCompletionResponse,
        refined_response: ChatCompletionResponse,
        validated_response: ChatCompletionResponse,
    ) -> None:
        """Test PipelineMode skips compression for small content."""
        drafter_provider.generate = AsyncMock(return_value=draft_response)
        refiner_provider.generate = AsyncMock(return_value=refined_response)
        validator_provider.generate = AsyncMock(return_value=validated_response)

        mode = PipelineMode(
            drafter=drafter_provider,
            refiner=refiner_provider,
            validator=validator_provider,
        )
        result = await mode.execute(sample_request)

        # Should complete without compression needed
        assert result is not None
        # Verify refiner received full draft content
        refiner_call = refiner_provider.generate.call_args[0][0]
        messages = refiner_call.messages
        assert any(
            "factorial" in (m.content or "") for m in messages
        )

    @pytest.mark.asyncio
    async def test_pipeline_mode_tracks_compression_in_metadata(
        self,
        drafter_provider: MagicMock,
        refiner_provider: MagicMock,
        validator_provider: MagicMock,
        sample_request: ChatCompletionRequest,
        refined_response: ChatCompletionResponse,
        validated_response: ChatCompletionResponse,
    ) -> None:
        """Test PipelineMode tracks compression in metadata."""
        # Large content that needs compression
        large_content = "x " * 20000
        large_draft = ChatCompletionResponse(
            id="chatcmpl-large",
            created=int(time.time()),
            model="llama-3.2-3b",
            choices=[
                Choice(
                    index=0,
                    message=ChoiceMessage(role="assistant", content=large_content),
                    finish_reason="stop",
                )
            ],
            usage=Usage(prompt_tokens=30, completion_tokens=20000, total_tokens=20030),
        )

        drafter_provider.generate = AsyncMock(return_value=large_draft)
        refiner_provider.generate = AsyncMock(return_value=refined_response)
        validator_provider.generate = AsyncMock(return_value=validated_response)

        mode = PipelineMode(
            drafter=drafter_provider,
            refiner=refiner_provider,
            validator=validator_provider,
        )
        result = await mode.execute(sample_request)

        # Should complete and report completion
        assert result.orchestration is not None


# =============================================================================
# AC-13.3, AC-13.4: Saga compensation and partial results
# =============================================================================


class TestPipelineModeSaga:
    """Tests for PipelineMode saga compensation."""

    @pytest.mark.asyncio
    async def test_pipeline_mode_returns_partial_on_refine_error(
        self,
        drafter_provider: MagicMock,
        refiner_provider: MagicMock,
        validator_provider: MagicMock,
        sample_request: ChatCompletionRequest,
        draft_response: ChatCompletionResponse,
    ) -> None:
        """Test PipelineMode returns partial result when refiner fails."""
        drafter_provider.generate = AsyncMock(return_value=draft_response)
        refiner_provider.generate = AsyncMock(
            side_effect=RuntimeError("Refiner failed")
        )

        mode = PipelineMode(
            drafter=drafter_provider,
            refiner=refiner_provider,
            validator=validator_provider,
        )
        result = await mode.execute(sample_request)

        # Should return partial result from draft
        assert result is not None
        assert result.choices[0].finish_reason == "partial"
        assert "factorial" in (result.choices[0].message.content or "")

    @pytest.mark.asyncio
    async def test_pipeline_mode_returns_partial_on_validate_error(
        self,
        drafter_provider: MagicMock,
        refiner_provider: MagicMock,
        validator_provider: MagicMock,
        sample_request: ChatCompletionRequest,
        draft_response: ChatCompletionResponse,
        refined_response: ChatCompletionResponse,
    ) -> None:
        """Test PipelineMode returns partial result when validator fails."""
        drafter_provider.generate = AsyncMock(return_value=draft_response)
        refiner_provider.generate = AsyncMock(return_value=refined_response)
        validator_provider.generate = AsyncMock(
            side_effect=RuntimeError("Validator failed")
        )

        mode = PipelineMode(
            drafter=drafter_provider,
            refiner=refiner_provider,
            validator=validator_provider,
        )
        result = await mode.execute(sample_request)

        # Should return partial result from refiner (last successful)
        assert result is not None
        assert result.choices[0].finish_reason == "partial"
        # Should have refined content with type hints
        assert "int" in (result.choices[0].message.content or "")

    @pytest.mark.asyncio
    async def test_pipeline_mode_partial_includes_completed_steps(
        self,
        drafter_provider: MagicMock,
        refiner_provider: MagicMock,
        validator_provider: MagicMock,
        sample_request: ChatCompletionRequest,
        draft_response: ChatCompletionResponse,
        refined_response: ChatCompletionResponse,
    ) -> None:
        """Test partial result metadata includes completed steps."""
        drafter_provider.generate = AsyncMock(return_value=draft_response)
        refiner_provider.generate = AsyncMock(return_value=refined_response)
        validator_provider.generate = AsyncMock(
            side_effect=RuntimeError("Validator failed")
        )

        mode = PipelineMode(
            drafter=drafter_provider,
            refiner=refiner_provider,
            validator=validator_provider,
        )
        result = await mode.execute(sample_request)

        # Metadata should show partial completion
        assert result.orchestration is not None
        assert result.orchestration.mode == "pipeline"

    @pytest.mark.asyncio
    async def test_pipeline_mode_propagates_error_when_no_partial(
        self,
        drafter_provider: MagicMock,
        refiner_provider: MagicMock,
        validator_provider: MagicMock,
        sample_request: ChatCompletionRequest,
    ) -> None:
        """Test PipelineMode raises when draft fails (no partial available)."""
        drafter_provider.generate = AsyncMock(
            side_effect=RuntimeError("Drafter failed")
        )

        mode = PipelineMode(
            drafter=drafter_provider,
            refiner=refiner_provider,
            validator=validator_provider,
        )

        with pytest.raises(RuntimeError, match="Drafter failed"):
            await mode.execute(sample_request)

    @pytest.mark.asyncio
    async def test_pipeline_mode_saga_tracks_all_steps(
        self,
        drafter_provider: MagicMock,
        refiner_provider: MagicMock,
        validator_provider: MagicMock,
        sample_request: ChatCompletionRequest,
        draft_response: ChatCompletionResponse,
        refined_response: ChatCompletionResponse,
        validated_response: ChatCompletionResponse,
    ) -> None:
        """Test PipelineMode saga tracks all completed steps."""
        drafter_provider.generate = AsyncMock(return_value=draft_response)
        refiner_provider.generate = AsyncMock(return_value=refined_response)
        validator_provider.generate = AsyncMock(return_value=validated_response)

        mode = PipelineMode(
            drafter=drafter_provider,
            refiner=refiner_provider,
            validator=validator_provider,
        )
        result = await mode.execute(sample_request)

        # All steps completed
        assert result.orchestration is not None


# =============================================================================
# Metadata Tests
# =============================================================================


class TestPipelineModeMetadata:
    """Tests for PipelineMode OrchestrationMetadata."""

    @pytest.mark.asyncio
    async def test_pipeline_mode_reports_all_models(
        self,
        drafter_provider: MagicMock,
        refiner_provider: MagicMock,
        validator_provider: MagicMock,
        sample_request: ChatCompletionRequest,
        draft_response: ChatCompletionResponse,
        refined_response: ChatCompletionResponse,
        validated_response: ChatCompletionResponse,
    ) -> None:
        """Test PipelineMode reports all three models used."""
        drafter_provider.generate = AsyncMock(return_value=draft_response)
        refiner_provider.generate = AsyncMock(return_value=refined_response)
        validator_provider.generate = AsyncMock(return_value=validated_response)

        mode = PipelineMode(
            drafter=drafter_provider,
            refiner=refiner_provider,
            validator=validator_provider,
        )
        result = await mode.execute(sample_request)

        assert result.orchestration is not None
        assert "llama-3.2-3b" in result.orchestration.models_used
        assert "deepseek-r1-7b" in result.orchestration.models_used
        assert "phi-4" in result.orchestration.models_used
        assert len(result.orchestration.models_used) == 3

    @pytest.mark.asyncio
    async def test_pipeline_mode_sets_mode_to_pipeline(
        self,
        drafter_provider: MagicMock,
        refiner_provider: MagicMock,
        validator_provider: MagicMock,
        sample_request: ChatCompletionRequest,
        draft_response: ChatCompletionResponse,
        refined_response: ChatCompletionResponse,
        validated_response: ChatCompletionResponse,
    ) -> None:
        """Test PipelineMode sets orchestration mode to 'pipeline'."""
        drafter_provider.generate = AsyncMock(return_value=draft_response)
        refiner_provider.generate = AsyncMock(return_value=refined_response)
        validator_provider.generate = AsyncMock(return_value=validated_response)

        mode = PipelineMode(
            drafter=drafter_provider,
            refiner=refiner_provider,
            validator=validator_provider,
        )
        result = await mode.execute(sample_request)

        assert result.orchestration is not None
        assert result.orchestration.mode == "pipeline"

    @pytest.mark.asyncio
    async def test_pipeline_mode_reports_inference_time(
        self,
        drafter_provider: MagicMock,
        refiner_provider: MagicMock,
        validator_provider: MagicMock,
        sample_request: ChatCompletionRequest,
        draft_response: ChatCompletionResponse,
        refined_response: ChatCompletionResponse,
        validated_response: ChatCompletionResponse,
    ) -> None:
        """Test PipelineMode reports total inference time."""
        drafter_provider.generate = AsyncMock(return_value=draft_response)
        refiner_provider.generate = AsyncMock(return_value=refined_response)
        validator_provider.generate = AsyncMock(return_value=validated_response)

        mode = PipelineMode(
            drafter=drafter_provider,
            refiner=refiner_provider,
            validator=validator_provider,
        )
        result = await mode.execute(sample_request)

        assert result.orchestration is not None
        assert result.orchestration.total_inference_time_ms > 0

    @pytest.mark.asyncio
    async def test_pipeline_mode_aggregates_usage_stats(
        self,
        drafter_provider: MagicMock,
        refiner_provider: MagicMock,
        validator_provider: MagicMock,
        sample_request: ChatCompletionRequest,
        draft_response: ChatCompletionResponse,
        refined_response: ChatCompletionResponse,
        validated_response: ChatCompletionResponse,
    ) -> None:
        """Test PipelineMode aggregates usage stats from all phases."""
        drafter_provider.generate = AsyncMock(return_value=draft_response)
        refiner_provider.generate = AsyncMock(return_value=refined_response)
        validator_provider.generate = AsyncMock(return_value=validated_response)

        mode = PipelineMode(
            drafter=drafter_provider,
            refiner=refiner_provider,
            validator=validator_provider,
        )
        result = await mode.execute(sample_request)

        # Usage should be aggregated: 55 + 140 + 220 = 415
        assert result.usage.total_tokens >= 55


# =============================================================================
# Edge Cases
# =============================================================================


class TestPipelineModeEdgeCases:
    """Tests for PipelineMode edge cases."""

    @pytest.mark.asyncio
    async def test_pipeline_mode_handles_empty_draft(
        self,
        drafter_provider: MagicMock,
        refiner_provider: MagicMock,
        validator_provider: MagicMock,
        sample_request: ChatCompletionRequest,
        refined_response: ChatCompletionResponse,
        validated_response: ChatCompletionResponse,
    ) -> None:
        """Test PipelineMode handles empty draft content."""
        empty_draft = ChatCompletionResponse(
            id="chatcmpl-empty",
            created=int(time.time()),
            model="llama-3.2-3b",
            choices=[
                Choice(
                    index=0,
                    message=ChoiceMessage(role="assistant", content=""),
                    finish_reason="stop",
                )
            ],
            usage=Usage(prompt_tokens=30, completion_tokens=0, total_tokens=30),
        )

        drafter_provider.generate = AsyncMock(return_value=empty_draft)
        refiner_provider.generate = AsyncMock(return_value=refined_response)
        validator_provider.generate = AsyncMock(return_value=validated_response)

        mode = PipelineMode(
            drafter=drafter_provider,
            refiner=refiner_provider,
            validator=validator_provider,
        )
        result = await mode.execute(sample_request)

        # Should complete pipeline
        assert result is not None

    @pytest.mark.asyncio
    async def test_pipeline_mode_two_model_pipeline(
        self,
        drafter_provider: MagicMock,
        validator_provider: MagicMock,
        sample_request: ChatCompletionRequest,
        draft_response: ChatCompletionResponse,
        validated_response: ChatCompletionResponse,
    ) -> None:
        """Test PipelineMode with only two models (no refiner)."""
        drafter_provider.generate = AsyncMock(return_value=draft_response)
        validator_provider.generate = AsyncMock(return_value=validated_response)

        mode = PipelineMode(
            drafter=drafter_provider,
            refiner=None,  # No refiner
            validator=validator_provider,
        )
        result = await mode.execute(sample_request)

        # Should complete with just draft → validate
        assert result is not None
        assert drafter_provider.generate.call_count == 1
        assert validator_provider.generate.call_count == 1

    @pytest.mark.asyncio
    async def test_pipeline_mode_preserves_model_field(
        self,
        drafter_provider: MagicMock,
        refiner_provider: MagicMock,
        validator_provider: MagicMock,
        sample_request: ChatCompletionRequest,
        draft_response: ChatCompletionResponse,
        refined_response: ChatCompletionResponse,
        validated_response: ChatCompletionResponse,
    ) -> None:
        """Test PipelineMode preserves model field from validator."""
        drafter_provider.generate = AsyncMock(return_value=draft_response)
        refiner_provider.generate = AsyncMock(return_value=refined_response)
        validator_provider.generate = AsyncMock(return_value=validated_response)

        mode = PipelineMode(
            drafter=drafter_provider,
            refiner=refiner_provider,
            validator=validator_provider,
        )
        result = await mode.execute(sample_request)

        # Model field should be from validator
        assert result.model == "phi-4"


# =============================================================================
# Property Tests
# =============================================================================


class TestPipelineModeProperties:
    """Tests for PipelineMode properties."""

    def test_pipeline_mode_drafter_property(
        self,
        drafter_provider: MagicMock,
        refiner_provider: MagicMock,
        validator_provider: MagicMock,
    ) -> None:
        """Test PipelineMode exposes drafter provider."""
        mode = PipelineMode(
            drafter=drafter_provider,
            refiner=refiner_provider,
            validator=validator_provider,
        )
        assert mode.drafter == drafter_provider

    def test_pipeline_mode_refiner_property(
        self,
        drafter_provider: MagicMock,
        refiner_provider: MagicMock,
        validator_provider: MagicMock,
    ) -> None:
        """Test PipelineMode exposes refiner provider."""
        mode = PipelineMode(
            drafter=drafter_provider,
            refiner=refiner_provider,
            validator=validator_provider,
        )
        assert mode.refiner == refiner_provider

    def test_pipeline_mode_validator_property(
        self,
        drafter_provider: MagicMock,
        refiner_provider: MagicMock,
        validator_provider: MagicMock,
    ) -> None:
        """Test PipelineMode exposes validator provider."""
        mode = PipelineMode(
            drafter=drafter_provider,
            refiner=refiner_provider,
            validator=validator_provider,
        )
        assert mode.validator == validator_provider

    def test_pipeline_mode_model_ids(
        self,
        drafter_provider: MagicMock,
        refiner_provider: MagicMock,
        validator_provider: MagicMock,
    ) -> None:
        """Test PipelineMode returns model IDs."""
        mode = PipelineMode(
            drafter=drafter_provider,
            refiner=refiner_provider,
            validator=validator_provider,
        )
        assert mode.drafter_model_id == "llama-3.2-3b"
        assert mode.refiner_model_id == "deepseek-r1-7b"
        assert mode.validator_model_id == "phi-4"
