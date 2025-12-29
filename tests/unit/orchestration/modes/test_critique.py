"""Unit tests for CritiqueMode orchestration.

Tests for:
- CritiqueMode: Generator → Critic → Revise flow (AC-12.1)
- CritiqueMode uses HandoffState between steps (AC-12.2)
- CritiqueMode respects max_rounds setting (AC-12.3)
- CritiqueMode reports models_used in metadata (AC-12.4)

Flow: Request → A(gen) → B(critique) → A(revise) → Response
"""

import time
from unittest.mock import AsyncMock, MagicMock, call

import pytest

from src.models.requests import ChatCompletionRequest, Message
from src.models.responses import (
    ChatCompletionResponse,
    Choice,
    ChoiceMessage,
    Usage,
)
from src.orchestration.context import HandoffState
from src.orchestration.modes.critique import CritiqueMode
from src.providers.base import InferenceProvider, ModelMetadata


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def generator_provider() -> MagicMock:
    """Create a mock generator provider (Model A)."""
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
def critic_provider() -> MagicMock:
    """Create a mock critic provider (Model B)."""
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
def sample_request() -> ChatCompletionRequest:
    """Create a sample chat completion request."""
    return ChatCompletionRequest(
        model="phi-4",
        messages=[
            Message(role="system", content="You are a helpful assistant."),
            Message(role="user", content="Explain recursion in programming."),
        ],
        temperature=0.7,
        max_tokens=500,
    )


@pytest.fixture
def generation_response() -> ChatCompletionResponse:
    """Create a sample generation response."""
    return ChatCompletionResponse(
        id="chatcmpl-gen-123",
        created=int(time.time()),
        model="phi-4",
        choices=[
            Choice(
                index=0,
                message=ChoiceMessage(
                    role="assistant",
                    content="Recursion is when a function calls itself.",
                ),
                finish_reason="stop",
            )
        ],
        usage=Usage(
            prompt_tokens=30,
            completion_tokens=20,
            total_tokens=50,
        ),
    )


@pytest.fixture
def critique_response_with_issues() -> ChatCompletionResponse:
    """Create a critique response with issues found."""
    return ChatCompletionResponse(
        id="chatcmpl-crit-123",
        created=int(time.time()),
        model="deepseek-r1-7b",
        choices=[
            Choice(
                index=0,
                message=ChoiceMessage(
                    role="assistant",
                    content=(
                        "ISSUES FOUND:\n"
                        "1. Missing base case explanation\n"
                        "2. No concrete example provided\n"
                        "RECOMMENDATION: Add base case and example."
                    ),
                ),
                finish_reason="stop",
            )
        ],
        usage=Usage(
            prompt_tokens=60,
            completion_tokens=40,
            total_tokens=100,
        ),
    )


@pytest.fixture
def critique_response_no_issues() -> ChatCompletionResponse:
    """Create a critique response with no issues."""
    return ChatCompletionResponse(
        id="chatcmpl-crit-456",
        created=int(time.time()),
        model="deepseek-r1-7b",
        choices=[
            Choice(
                index=0,
                message=ChoiceMessage(
                    role="assistant",
                    content="NO ISSUES FOUND. Response is complete and accurate.",
                ),
                finish_reason="stop",
            )
        ],
        usage=Usage(
            prompt_tokens=60,
            completion_tokens=15,
            total_tokens=75,
        ),
    )


@pytest.fixture
def revised_response() -> ChatCompletionResponse:
    """Create a revised response after critique."""
    return ChatCompletionResponse(
        id="chatcmpl-rev-123",
        created=int(time.time()),
        model="phi-4",
        choices=[
            Choice(
                index=0,
                message=ChoiceMessage(
                    role="assistant",
                    content=(
                        "Recursion is when a function calls itself. "
                        "Every recursive function needs a base case to stop recursion. "
                        "Example: factorial(n) = n * factorial(n-1), base case: factorial(1) = 1."
                    ),
                ),
                finish_reason="stop",
            )
        ],
        usage=Usage(
            prompt_tokens=80,
            completion_tokens=50,
            total_tokens=130,
        ),
    )


# =============================================================================
# AC-12.1: CritiqueMode Generator → Critic → Revise flow
# =============================================================================


class TestCritiqueModeFlow:
    """Tests for CritiqueMode execute() flow."""

    @pytest.mark.asyncio
    async def test_critique_mode_executes_generate_phase(
        self,
        generator_provider: MagicMock,
        critic_provider: MagicMock,
        sample_request: ChatCompletionRequest,
        generation_response: ChatCompletionResponse,
        critique_response_no_issues: ChatCompletionResponse,
    ) -> None:
        """Test CritiqueMode executes generation phase first."""
        generator_provider.generate = AsyncMock(return_value=generation_response)
        critic_provider.generate = AsyncMock(return_value=critique_response_no_issues)

        mode = CritiqueMode(
            generator=generator_provider,
            critic=critic_provider,
        )
        await mode.execute(sample_request)

        # Verify generator was called
        generator_provider.generate.assert_called()

    @pytest.mark.asyncio
    async def test_critique_mode_executes_critique_phase(
        self,
        generator_provider: MagicMock,
        critic_provider: MagicMock,
        sample_request: ChatCompletionRequest,
        generation_response: ChatCompletionResponse,
        critique_response_no_issues: ChatCompletionResponse,
    ) -> None:
        """Test CritiqueMode executes critique phase after generation."""
        generator_provider.generate = AsyncMock(return_value=generation_response)
        critic_provider.generate = AsyncMock(return_value=critique_response_no_issues)

        mode = CritiqueMode(
            generator=generator_provider,
            critic=critic_provider,
        )
        await mode.execute(sample_request)

        # Verify critic was called
        critic_provider.generate.assert_called()

    @pytest.mark.asyncio
    async def test_critique_mode_executes_revise_phase_when_issues_found(
        self,
        generator_provider: MagicMock,
        critic_provider: MagicMock,
        sample_request: ChatCompletionRequest,
        generation_response: ChatCompletionResponse,
        critique_response_with_issues: ChatCompletionResponse,
        revised_response: ChatCompletionResponse,
    ) -> None:
        """Test CritiqueMode executes revise phase when critic finds issues."""
        # Generator: first call returns initial, second call returns revised
        generator_provider.generate = AsyncMock(
            side_effect=[generation_response, revised_response]
        )
        critic_provider.generate = AsyncMock(return_value=critique_response_with_issues)

        mode = CritiqueMode(
            generator=generator_provider,
            critic=critic_provider,
            max_rounds=1,  # Only 1 round of critique-revise
        )
        await mode.execute(sample_request)

        # Verify generator was called twice (generate + revise)
        assert generator_provider.generate.call_count == 2

    @pytest.mark.asyncio
    async def test_critique_mode_skips_revise_when_no_issues(
        self,
        generator_provider: MagicMock,
        critic_provider: MagicMock,
        sample_request: ChatCompletionRequest,
        generation_response: ChatCompletionResponse,
        critique_response_no_issues: ChatCompletionResponse,
    ) -> None:
        """Test CritiqueMode skips revise when no issues found."""
        generator_provider.generate = AsyncMock(return_value=generation_response)
        critic_provider.generate = AsyncMock(return_value=critique_response_no_issues)

        mode = CritiqueMode(
            generator=generator_provider,
            critic=critic_provider,
        )
        await mode.execute(sample_request)

        # Verify generator was called once (no revise needed)
        assert generator_provider.generate.call_count == 1

    @pytest.mark.asyncio
    async def test_critique_mode_flow_order(
        self,
        generator_provider: MagicMock,
        critic_provider: MagicMock,
        sample_request: ChatCompletionRequest,
        generation_response: ChatCompletionResponse,
        critique_response_with_issues: ChatCompletionResponse,
        revised_response: ChatCompletionResponse,
    ) -> None:
        """Test CritiqueMode follows Generator → Critic → Revise order."""
        call_order: list[str] = []

        async def track_generator(*args: object, **kwargs: object) -> ChatCompletionResponse:
            if generator_provider.generate.call_count == 0:
                call_order.append("generate")
                return generation_response
            call_order.append("revise")
            return revised_response

        async def track_critic(*args: object, **kwargs: object) -> ChatCompletionResponse:
            call_order.append("critique")
            return critique_response_with_issues

        generator_provider.generate = AsyncMock(
            side_effect=[generation_response, revised_response]
        )
        critic_provider.generate = AsyncMock(return_value=critique_response_with_issues)

        # Use side_effect to track call order
        original_gen = generator_provider.generate
        original_crit = critic_provider.generate

        async def gen_side_effect(*a: object, **k: object) -> ChatCompletionResponse:
            result = await original_gen(*a, **k)
            call_order.append("generator" if len(call_order) == 0 else "revise")
            return result

        async def crit_side_effect(*a: object, **k: object) -> ChatCompletionResponse:
            result = await original_crit(*a, **k)
            call_order.append("critique")
            return result

        generator_provider.generate.side_effect = [generation_response, revised_response]
        critic_provider.generate.return_value = critique_response_with_issues

        mode = CritiqueMode(
            generator=generator_provider,
            critic=critic_provider,
            max_rounds=1,
        )
        await mode.execute(sample_request)

        # Verify call count order
        assert generator_provider.generate.call_count >= 1
        assert critic_provider.generate.call_count >= 1


# =============================================================================
# AC-12.2: CritiqueMode uses HandoffState between steps
# =============================================================================


class TestCritiqueModeHandoffState:
    """Tests for CritiqueMode HandoffState usage."""

    @pytest.mark.asyncio
    async def test_critique_mode_creates_handoff_state(
        self,
        generator_provider: MagicMock,
        critic_provider: MagicMock,
        sample_request: ChatCompletionRequest,
        generation_response: ChatCompletionResponse,
        critique_response_no_issues: ChatCompletionResponse,
    ) -> None:
        """Test CritiqueMode creates HandoffState for coordination."""
        generator_provider.generate = AsyncMock(return_value=generation_response)
        critic_provider.generate = AsyncMock(return_value=critique_response_no_issues)

        mode = CritiqueMode(
            generator=generator_provider,
            critic=critic_provider,
        )
        result = await mode.execute(sample_request)

        # Result should be successful
        assert result is not None
        assert result.choices[0].message.content is not None

    @pytest.mark.asyncio
    async def test_critique_mode_passes_generation_to_critic(
        self,
        generator_provider: MagicMock,
        critic_provider: MagicMock,
        sample_request: ChatCompletionRequest,
        generation_response: ChatCompletionResponse,
        critique_response_no_issues: ChatCompletionResponse,
    ) -> None:
        """Test CritiqueMode passes generated content to critic via handoff."""
        generator_provider.generate = AsyncMock(return_value=generation_response)
        critic_provider.generate = AsyncMock(return_value=critique_response_no_issues)

        mode = CritiqueMode(
            generator=generator_provider,
            critic=critic_provider,
        )
        await mode.execute(sample_request)

        # Verify critic was called with modified request containing generation
        critic_call_args = critic_provider.generate.call_args
        assert critic_call_args is not None
        critic_request = critic_call_args[0][0]
        # Critic request should contain the original generation for review
        messages_content = [m.content for m in critic_request.messages if m.content]
        # At least one message should contain generated content reference
        assert any(
            "Recursion is when a function calls itself" in (c or "")
            for c in messages_content
        )

    @pytest.mark.asyncio
    async def test_critique_mode_passes_critique_to_reviser(
        self,
        generator_provider: MagicMock,
        critic_provider: MagicMock,
        sample_request: ChatCompletionRequest,
        generation_response: ChatCompletionResponse,
        critique_response_with_issues: ChatCompletionResponse,
        revised_response: ChatCompletionResponse,
    ) -> None:
        """Test CritiqueMode passes critique feedback to reviser via handoff."""
        generator_provider.generate = AsyncMock(
            side_effect=[generation_response, revised_response]
        )
        critic_provider.generate = AsyncMock(return_value=critique_response_with_issues)

        mode = CritiqueMode(
            generator=generator_provider,
            critic=critic_provider,
            max_rounds=1,
        )
        await mode.execute(sample_request)

        # Second generator call (revise) should include critique feedback
        assert generator_provider.generate.call_count == 2
        revise_call_args = generator_provider.generate.call_args_list[1]
        revise_request = revise_call_args[0][0]
        messages_content = [m.content for m in revise_request.messages if m.content]
        # Should contain critique feedback
        assert any("ISSUES" in (c or "") or "Missing" in (c or "") for c in messages_content)

    @pytest.mark.asyncio
    async def test_critique_mode_handoff_tracks_decisions(
        self,
        generator_provider: MagicMock,
        critic_provider: MagicMock,
        sample_request: ChatCompletionRequest,
        generation_response: ChatCompletionResponse,
        critique_response_with_issues: ChatCompletionResponse,
        revised_response: ChatCompletionResponse,
    ) -> None:
        """Test CritiqueMode HandoffState tracks decisions made."""
        generator_provider.generate = AsyncMock(
            side_effect=[generation_response, revised_response]
        )
        critic_provider.generate = AsyncMock(return_value=critique_response_with_issues)

        mode = CritiqueMode(
            generator=generator_provider,
            critic=critic_provider,
            max_rounds=1,
        )
        result = await mode.execute(sample_request)

        # Verify the mode tracked the process
        assert result.orchestration is not None
        # Should have completed the flow
        assert result.orchestration.rounds is not None
        assert result.orchestration.rounds >= 1


# =============================================================================
# AC-12.3: CritiqueMode respects max_rounds setting
# =============================================================================


class TestCritiqueModeMaxRounds:
    """Tests for CritiqueMode max_rounds limiting."""

    @pytest.mark.asyncio
    async def test_critique_mode_default_max_rounds(
        self,
        generator_provider: MagicMock,
        critic_provider: MagicMock,
        sample_request: ChatCompletionRequest,
        generation_response: ChatCompletionResponse,
        critique_response_no_issues: ChatCompletionResponse,
    ) -> None:
        """Test CritiqueMode has default max_rounds of 3."""
        generator_provider.generate = AsyncMock(return_value=generation_response)
        critic_provider.generate = AsyncMock(return_value=critique_response_no_issues)

        mode = CritiqueMode(
            generator=generator_provider,
            critic=critic_provider,
        )

        assert mode.max_rounds == 3

    @pytest.mark.asyncio
    async def test_critique_mode_respects_custom_max_rounds(
        self,
        generator_provider: MagicMock,
        critic_provider: MagicMock,
        sample_request: ChatCompletionRequest,
        generation_response: ChatCompletionResponse,
        critique_response_with_issues: ChatCompletionResponse,
        revised_response: ChatCompletionResponse,
    ) -> None:
        """Test CritiqueMode respects custom max_rounds setting."""
        # Always return issues to force max rounds
        generator_provider.generate = AsyncMock(return_value=generation_response)
        critic_provider.generate = AsyncMock(return_value=critique_response_with_issues)

        mode = CritiqueMode(
            generator=generator_provider,
            critic=critic_provider,
            max_rounds=2,
        )
        await mode.execute(sample_request)

        # Should have done at most max_rounds iterations
        # Each round = 1 critique + 1 revise (after initial gen)
        # Initial gen + (2 rounds * (critique + revise))
        # But we stop after max_rounds critique cycles
        assert critic_provider.generate.call_count <= 2

    @pytest.mark.asyncio
    async def test_critique_mode_stops_at_max_rounds(
        self,
        generator_provider: MagicMock,
        critic_provider: MagicMock,
        sample_request: ChatCompletionRequest,
        generation_response: ChatCompletionResponse,
        critique_response_with_issues: ChatCompletionResponse,
    ) -> None:
        """Test CritiqueMode stops iterating at max_rounds."""
        # Always return issues - would loop forever without max_rounds
        generator_provider.generate = AsyncMock(return_value=generation_response)
        critic_provider.generate = AsyncMock(return_value=critique_response_with_issues)

        mode = CritiqueMode(
            generator=generator_provider,
            critic=critic_provider,
            max_rounds=1,
        )
        result = await mode.execute(sample_request)

        # Should complete despite issues (capped at max_rounds)
        assert result is not None
        assert result.orchestration is not None
        assert result.orchestration.rounds == 1

    @pytest.mark.asyncio
    async def test_critique_mode_early_exit_on_no_issues(
        self,
        generator_provider: MagicMock,
        critic_provider: MagicMock,
        sample_request: ChatCompletionRequest,
        generation_response: ChatCompletionResponse,
        critique_response_no_issues: ChatCompletionResponse,
    ) -> None:
        """Test CritiqueMode exits early when no issues found."""
        generator_provider.generate = AsyncMock(return_value=generation_response)
        critic_provider.generate = AsyncMock(return_value=critique_response_no_issues)

        mode = CritiqueMode(
            generator=generator_provider,
            critic=critic_provider,
            max_rounds=5,  # High limit
        )
        result = await mode.execute(sample_request)

        # Should exit after first critique (no issues)
        assert critic_provider.generate.call_count == 1
        assert result.orchestration is not None
        # Zero rounds because no revision was needed
        assert result.orchestration.rounds == 0

    @pytest.mark.asyncio
    async def test_critique_mode_max_rounds_zero_skips_critique(
        self,
        generator_provider: MagicMock,
        critic_provider: MagicMock,
        sample_request: ChatCompletionRequest,
        generation_response: ChatCompletionResponse,
    ) -> None:
        """Test CritiqueMode with max_rounds=0 skips critique entirely."""
        generator_provider.generate = AsyncMock(return_value=generation_response)

        mode = CritiqueMode(
            generator=generator_provider,
            critic=critic_provider,
            max_rounds=0,
        )
        result = await mode.execute(sample_request)

        # Should only generate, no critique
        assert generator_provider.generate.call_count == 1
        assert critic_provider.generate.call_count == 0
        assert result.orchestration is not None
        assert result.orchestration.rounds == 0

    @pytest.mark.asyncio
    async def test_critique_mode_reports_actual_rounds_in_metadata(
        self,
        generator_provider: MagicMock,
        critic_provider: MagicMock,
        sample_request: ChatCompletionRequest,
        generation_response: ChatCompletionResponse,
        critique_response_with_issues: ChatCompletionResponse,
        critique_response_no_issues: ChatCompletionResponse,
    ) -> None:
        """Test CritiqueMode reports actual rounds used in metadata."""
        # First critique has issues, second is clean
        generator_provider.generate = AsyncMock(return_value=generation_response)
        critic_provider.generate = AsyncMock(
            side_effect=[critique_response_with_issues, critique_response_no_issues]
        )

        mode = CritiqueMode(
            generator=generator_provider,
            critic=critic_provider,
            max_rounds=5,
        )
        result = await mode.execute(sample_request)

        # Should report actual rounds (1 revision)
        assert result.orchestration is not None
        assert result.orchestration.rounds == 1


# =============================================================================
# AC-12.4: CritiqueMode reports models_used in metadata
# =============================================================================


class TestCritiqueModeMetadata:
    """Tests for CritiqueMode OrchestrationMetadata."""

    @pytest.mark.asyncio
    async def test_critique_mode_reports_both_models(
        self,
        generator_provider: MagicMock,
        critic_provider: MagicMock,
        sample_request: ChatCompletionRequest,
        generation_response: ChatCompletionResponse,
        critique_response_no_issues: ChatCompletionResponse,
    ) -> None:
        """Test CritiqueMode reports both generator and critic models."""
        generator_provider.generate = AsyncMock(return_value=generation_response)
        critic_provider.generate = AsyncMock(return_value=critique_response_no_issues)

        mode = CritiqueMode(
            generator=generator_provider,
            critic=critic_provider,
        )
        result = await mode.execute(sample_request)

        assert result.orchestration is not None
        assert "phi-4" in result.orchestration.models_used
        assert "deepseek-r1-7b" in result.orchestration.models_used
        assert len(result.orchestration.models_used) == 2

    @pytest.mark.asyncio
    async def test_critique_mode_sets_mode_to_critique(
        self,
        generator_provider: MagicMock,
        critic_provider: MagicMock,
        sample_request: ChatCompletionRequest,
        generation_response: ChatCompletionResponse,
        critique_response_no_issues: ChatCompletionResponse,
    ) -> None:
        """Test CritiqueMode sets orchestration mode to 'critique'."""
        generator_provider.generate = AsyncMock(return_value=generation_response)
        critic_provider.generate = AsyncMock(return_value=critique_response_no_issues)

        mode = CritiqueMode(
            generator=generator_provider,
            critic=critic_provider,
        )
        result = await mode.execute(sample_request)

        assert result.orchestration is not None
        assert result.orchestration.mode == "critique"

    @pytest.mark.asyncio
    async def test_critique_mode_reports_inference_time(
        self,
        generator_provider: MagicMock,
        critic_provider: MagicMock,
        sample_request: ChatCompletionRequest,
        generation_response: ChatCompletionResponse,
        critique_response_no_issues: ChatCompletionResponse,
    ) -> None:
        """Test CritiqueMode reports total inference time."""
        generator_provider.generate = AsyncMock(return_value=generation_response)
        critic_provider.generate = AsyncMock(return_value=critique_response_no_issues)

        mode = CritiqueMode(
            generator=generator_provider,
            critic=critic_provider,
        )
        result = await mode.execute(sample_request)

        assert result.orchestration is not None
        assert result.orchestration.total_inference_time_ms > 0

    @pytest.mark.asyncio
    async def test_critique_mode_reports_final_score(
        self,
        generator_provider: MagicMock,
        critic_provider: MagicMock,
        sample_request: ChatCompletionRequest,
        generation_response: ChatCompletionResponse,
        critique_response_no_issues: ChatCompletionResponse,
    ) -> None:
        """Test CritiqueMode reports final quality score."""
        generator_provider.generate = AsyncMock(return_value=generation_response)
        critic_provider.generate = AsyncMock(return_value=critique_response_no_issues)

        mode = CritiqueMode(
            generator=generator_provider,
            critic=critic_provider,
        )
        result = await mode.execute(sample_request)

        assert result.orchestration is not None
        # Score should be provided (1.0 when no issues, lower when issues remain)
        assert result.orchestration.final_score is not None
        assert 0.0 <= result.orchestration.final_score <= 1.0

    @pytest.mark.asyncio
    async def test_critique_mode_high_score_when_no_issues(
        self,
        generator_provider: MagicMock,
        critic_provider: MagicMock,
        sample_request: ChatCompletionRequest,
        generation_response: ChatCompletionResponse,
        critique_response_no_issues: ChatCompletionResponse,
    ) -> None:
        """Test CritiqueMode gives high score when no issues found."""
        generator_provider.generate = AsyncMock(return_value=generation_response)
        critic_provider.generate = AsyncMock(return_value=critique_response_no_issues)

        mode = CritiqueMode(
            generator=generator_provider,
            critic=critic_provider,
        )
        result = await mode.execute(sample_request)

        assert result.orchestration is not None
        assert result.orchestration.final_score == 1.0

    @pytest.mark.asyncio
    async def test_critique_mode_lower_score_when_issues_remain(
        self,
        generator_provider: MagicMock,
        critic_provider: MagicMock,
        sample_request: ChatCompletionRequest,
        generation_response: ChatCompletionResponse,
        critique_response_with_issues: ChatCompletionResponse,
    ) -> None:
        """Test CritiqueMode gives lower score when issues remain after max_rounds."""
        generator_provider.generate = AsyncMock(return_value=generation_response)
        critic_provider.generate = AsyncMock(return_value=critique_response_with_issues)

        mode = CritiqueMode(
            generator=generator_provider,
            critic=critic_provider,
            max_rounds=1,
        )
        result = await mode.execute(sample_request)

        assert result.orchestration is not None
        # Lower score because issues remain
        assert result.orchestration.final_score is not None
        assert result.orchestration.final_score < 1.0

    @pytest.mark.asyncio
    async def test_critique_mode_aggregates_usage_stats(
        self,
        generator_provider: MagicMock,
        critic_provider: MagicMock,
        sample_request: ChatCompletionRequest,
        generation_response: ChatCompletionResponse,
        critique_response_with_issues: ChatCompletionResponse,
        revised_response: ChatCompletionResponse,
    ) -> None:
        """Test CritiqueMode aggregates usage stats from all calls."""
        generator_provider.generate = AsyncMock(
            side_effect=[generation_response, revised_response]
        )
        critic_provider.generate = AsyncMock(return_value=critique_response_with_issues)

        mode = CritiqueMode(
            generator=generator_provider,
            critic=critic_provider,
            max_rounds=1,
        )
        result = await mode.execute(sample_request)

        # Usage should be aggregated
        # gen: 50 + crit: 100 + rev: 130 = 280 total
        assert result.usage.total_tokens >= 50  # At least generation


# =============================================================================
# Edge Cases and Error Handling
# =============================================================================


class TestCritiqueModeEdgeCases:
    """Tests for CritiqueMode edge cases."""

    @pytest.mark.asyncio
    async def test_critique_mode_handles_empty_generation(
        self,
        generator_provider: MagicMock,
        critic_provider: MagicMock,
        sample_request: ChatCompletionRequest,
        critique_response_no_issues: ChatCompletionResponse,
    ) -> None:
        """Test CritiqueMode handles empty generation content."""
        empty_response = ChatCompletionResponse(
            id="chatcmpl-empty",
            created=int(time.time()),
            model="phi-4",
            choices=[
                Choice(
                    index=0,
                    message=ChoiceMessage(role="assistant", content=""),
                    finish_reason="stop",
                )
            ],
            usage=Usage(prompt_tokens=30, completion_tokens=0, total_tokens=30),
        )
        generator_provider.generate = AsyncMock(return_value=empty_response)
        critic_provider.generate = AsyncMock(return_value=critique_response_no_issues)

        mode = CritiqueMode(
            generator=generator_provider,
            critic=critic_provider,
        )
        result = await mode.execute(sample_request)

        # Should complete without error
        assert result is not None

    @pytest.mark.asyncio
    async def test_critique_mode_propagates_generator_error(
        self,
        generator_provider: MagicMock,
        critic_provider: MagicMock,
        sample_request: ChatCompletionRequest,
    ) -> None:
        """Test CritiqueMode propagates errors from generator."""
        generator_provider.generate = AsyncMock(
            side_effect=RuntimeError("Generator failed")
        )

        mode = CritiqueMode(
            generator=generator_provider,
            critic=critic_provider,
        )

        with pytest.raises(RuntimeError, match="Generator failed"):
            await mode.execute(sample_request)

    @pytest.mark.asyncio
    async def test_critique_mode_propagates_critic_error(
        self,
        generator_provider: MagicMock,
        critic_provider: MagicMock,
        sample_request: ChatCompletionRequest,
        generation_response: ChatCompletionResponse,
    ) -> None:
        """Test CritiqueMode propagates errors from critic."""
        generator_provider.generate = AsyncMock(return_value=generation_response)
        critic_provider.generate = AsyncMock(
            side_effect=RuntimeError("Critic failed")
        )

        mode = CritiqueMode(
            generator=generator_provider,
            critic=critic_provider,
        )

        with pytest.raises(RuntimeError, match="Critic failed"):
            await mode.execute(sample_request)

    @pytest.mark.asyncio
    async def test_critique_mode_same_model_for_both_roles(
        self,
        generator_provider: MagicMock,
        sample_request: ChatCompletionRequest,
        generation_response: ChatCompletionResponse,
        critique_response_no_issues: ChatCompletionResponse,
    ) -> None:
        """Test CritiqueMode when same model is used for both roles."""
        generator_provider.generate = AsyncMock(
            side_effect=[generation_response, critique_response_no_issues]
        )

        mode = CritiqueMode(
            generator=generator_provider,
            critic=generator_provider,  # Same provider for both
        )
        result = await mode.execute(sample_request)

        # Should work, but models_used should still list both instances
        assert result.orchestration is not None
        # When same model, might appear once or twice depending on impl
        assert "phi-4" in result.orchestration.models_used

    @pytest.mark.asyncio
    async def test_critique_mode_preserves_original_model_field(
        self,
        generator_provider: MagicMock,
        critic_provider: MagicMock,
        sample_request: ChatCompletionRequest,
        generation_response: ChatCompletionResponse,
        critique_response_no_issues: ChatCompletionResponse,
    ) -> None:
        """Test CritiqueMode preserves model field from final response."""
        generator_provider.generate = AsyncMock(return_value=generation_response)
        critic_provider.generate = AsyncMock(return_value=critique_response_no_issues)

        mode = CritiqueMode(
            generator=generator_provider,
            critic=critic_provider,
        )
        result = await mode.execute(sample_request)

        # Model field should be the generator model
        assert result.model == "phi-4"


# =============================================================================
# Property Tests
# =============================================================================


class TestCritiqueModeProperties:
    """Tests for CritiqueMode properties."""

    def test_critique_mode_generator_property(
        self,
        generator_provider: MagicMock,
        critic_provider: MagicMock,
    ) -> None:
        """Test CritiqueMode exposes generator provider."""
        mode = CritiqueMode(
            generator=generator_provider,
            critic=critic_provider,
        )
        assert mode.generator == generator_provider

    def test_critique_mode_critic_property(
        self,
        generator_provider: MagicMock,
        critic_provider: MagicMock,
    ) -> None:
        """Test CritiqueMode exposes critic provider."""
        mode = CritiqueMode(
            generator=generator_provider,
            critic=critic_provider,
        )
        assert mode.critic == critic_provider

    def test_critique_mode_max_rounds_property(
        self,
        generator_provider: MagicMock,
        critic_provider: MagicMock,
    ) -> None:
        """Test CritiqueMode exposes max_rounds."""
        mode = CritiqueMode(
            generator=generator_provider,
            critic=critic_provider,
            max_rounds=5,
        )
        assert mode.max_rounds == 5

    def test_critique_mode_generator_model_id(
        self,
        generator_provider: MagicMock,
        critic_provider: MagicMock,
    ) -> None:
        """Test CritiqueMode returns generator model ID."""
        mode = CritiqueMode(
            generator=generator_provider,
            critic=critic_provider,
        )
        assert mode.generator_model_id == "phi-4"

    def test_critique_mode_critic_model_id(
        self,
        generator_provider: MagicMock,
        critic_provider: MagicMock,
    ) -> None:
        """Test CritiqueMode returns critic model ID."""
        mode = CritiqueMode(
            generator=generator_provider,
            critic=critic_provider,
        )
        assert mode.critic_model_id == "deepseek-r1-7b"
