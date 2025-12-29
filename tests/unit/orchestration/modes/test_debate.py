"""Unit tests for DebateMode orchestration.

Tests DebateMode functionality including:
- Parallel generation with asyncio.gather (AC-14.1, AC-14.2)
- Output comparison for agreement percentage (AC-14.3)
- Reconciliation to synthesize final answer (AC-14.4)

Reference: WBS-INF14
"""

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.models.requests import ChatCompletionRequest, Message
from src.models.responses import (
    ChatCompletionResponse,
    Choice,
    ChoiceMessage,
    OrchestrationMetadata,
    Usage,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_request() -> ChatCompletionRequest:
    """Create a sample chat completion request."""
    return ChatCompletionRequest(
        model="test-model",
        messages=[
            Message(role="user", content="What is the capital of France?"),
        ],
    )


@pytest.fixture
def mock_response_a() -> ChatCompletionResponse:
    """Create mock response from Model A."""
    return ChatCompletionResponse(
        id="chatcmpl-response-a",
        created=1234567890,
        model="model-a",
        choices=[
            Choice(
                index=0,
                message=ChoiceMessage(
                    role="assistant",
                    content="The capital of France is Paris. Paris has been the capital since 987 AD.",
                ),
                finish_reason="stop",
            )
        ],
        usage=Usage(prompt_tokens=10, completion_tokens=20, total_tokens=30),
    )


@pytest.fixture
def mock_response_b() -> ChatCompletionResponse:
    """Create mock response from Model B."""
    return ChatCompletionResponse(
        id="chatcmpl-response-b",
        created=1234567890,
        model="model-b",
        choices=[
            Choice(
                index=0,
                message=ChoiceMessage(
                    role="assistant",
                    content="Paris is the capital of France. It is also the largest city in France.",
                ),
                finish_reason="stop",
            )
        ],
        usage=Usage(prompt_tokens=10, completion_tokens=18, total_tokens=28),
    )


@pytest.fixture
def mock_reconciled_response() -> ChatCompletionResponse:
    """Create mock reconciled response."""
    return ChatCompletionResponse(
        id="chatcmpl-reconciled",
        created=1234567890,
        model="reconciler-model",
        choices=[
            Choice(
                index=0,
                message=ChoiceMessage(
                    role="assistant",
                    content="The capital of France is Paris. Paris has been the capital since 987 AD and is also the largest city in France.",
                ),
                finish_reason="stop",
            )
        ],
        usage=Usage(prompt_tokens=50, completion_tokens=30, total_tokens=80),
    )


@pytest.fixture
def mock_provider_a(mock_response_a: ChatCompletionResponse) -> MagicMock:
    """Create mock provider A."""
    provider = MagicMock()
    provider.generate = AsyncMock(return_value=mock_response_a)
    provider.model_info = MagicMock()
    provider.model_info.model_id = "model-a"
    return provider


@pytest.fixture
def mock_provider_b(mock_response_b: ChatCompletionResponse) -> MagicMock:
    """Create mock provider B."""
    provider = MagicMock()
    provider.generate = AsyncMock(return_value=mock_response_b)
    provider.model_info = MagicMock()
    provider.model_info.model_id = "model-b"
    return provider


@pytest.fixture
def mock_reconciler(mock_reconciled_response: ChatCompletionResponse) -> MagicMock:
    """Create mock reconciler provider."""
    provider = MagicMock()
    provider.generate = AsyncMock(return_value=mock_reconciled_response)
    provider.model_info = MagicMock()
    provider.model_info.model_id = "reconciler-model"
    return provider


# =============================================================================
# Test: Parallel Generation (AC-14.1, AC-14.2)
# =============================================================================


class TestDebateModeParallelGeneration:
    """Tests for parallel generation in DebateMode."""

    async def test_debate_mode_calls_both_providers(
        self,
        mock_provider_a: MagicMock,
        mock_provider_b: MagicMock,
        mock_reconciler: MagicMock,
        sample_request: ChatCompletionRequest,
    ) -> None:
        """Both providers should be called during debate."""
        from src.orchestration.modes.debate import DebateMode

        mode = DebateMode(
            participant_a=mock_provider_a,
            participant_b=mock_provider_b,
            reconciler=mock_reconciler,
        )

        await mode.execute(sample_request)

        mock_provider_a.generate.assert_called_once()
        mock_provider_b.generate.assert_called_once()

    async def test_debate_mode_uses_asyncio_gather_for_parallel(
        self,
        mock_provider_a: MagicMock,
        mock_provider_b: MagicMock,
        mock_reconciler: MagicMock,
        sample_request: ChatCompletionRequest,
    ) -> None:
        """Providers should be called in parallel using asyncio.gather."""
        from src.orchestration.modes.debate import DebateMode

        # Track call order with timestamps
        call_times: list[tuple[str, float]] = []

        async def mock_generate_a(request: ChatCompletionRequest) -> ChatCompletionResponse:
            call_times.append(("a_start", time.perf_counter()))
            await asyncio.sleep(0.05)  # Simulate work
            call_times.append(("a_end", time.perf_counter()))
            return ChatCompletionResponse(
                id="a",
                created=0,
                model="a",
                choices=[
                    Choice(
                        index=0,
                        message=ChoiceMessage(role="assistant", content="Response A"),
                        finish_reason="stop",
                    )
                ],
                usage=Usage(prompt_tokens=10, completion_tokens=10, total_tokens=20),
            )

        async def mock_generate_b(request: ChatCompletionRequest) -> ChatCompletionResponse:
            call_times.append(("b_start", time.perf_counter()))
            await asyncio.sleep(0.05)  # Simulate work
            call_times.append(("b_end", time.perf_counter()))
            return ChatCompletionResponse(
                id="b",
                created=0,
                model="b",
                choices=[
                    Choice(
                        index=0,
                        message=ChoiceMessage(role="assistant", content="Response B"),
                        finish_reason="stop",
                    )
                ],
                usage=Usage(prompt_tokens=10, completion_tokens=10, total_tokens=20),
            )

        mock_provider_a.generate = mock_generate_a
        mock_provider_b.generate = mock_generate_b

        mode = DebateMode(
            participant_a=mock_provider_a,
            participant_b=mock_provider_b,
            reconciler=mock_reconciler,
        )

        start = time.perf_counter()
        await mode.execute(sample_request)
        elapsed = time.perf_counter() - start

        # Parallel execution should be faster than sequential (2 * 0.05s)
        # Allow some overhead, but should be < 0.15s (0.1 + 50% margin)
        assert elapsed < 0.15, f"Expected parallel execution, but took {elapsed}s"

        # Both should have started before either ended (parallel)
        start_times = [t for name, t in call_times if "start" in name]
        end_times = [t for name, t in call_times if "end" in name]
        assert min(end_times) > max(start_times), "Expected overlapping execution"

    async def test_debate_mode_handles_single_provider_error(
        self,
        mock_provider_a: MagicMock,
        mock_provider_b: MagicMock,
        mock_reconciler: MagicMock,
        sample_request: ChatCompletionRequest,
    ) -> None:
        """Should handle error from one provider gracefully."""
        from src.orchestration.modes.debate import DebateMode

        mock_provider_b.generate = AsyncMock(side_effect=Exception("Model B failed"))

        mode = DebateMode(
            participant_a=mock_provider_a,
            participant_b=mock_provider_b,
            reconciler=mock_reconciler,
        )

        # Should still return a response (from A) or raise appropriately
        with pytest.raises(Exception, match="Model B failed"):
            await mode.execute(sample_request)

    async def test_debate_mode_returns_chat_completion_response(
        self,
        mock_provider_a: MagicMock,
        mock_provider_b: MagicMock,
        mock_reconciler: MagicMock,
        sample_request: ChatCompletionRequest,
    ) -> None:
        """Should return a valid ChatCompletionResponse."""
        from src.orchestration.modes.debate import DebateMode

        mode = DebateMode(
            participant_a=mock_provider_a,
            participant_b=mock_provider_b,
            reconciler=mock_reconciler,
        )

        result = await mode.execute(sample_request)

        assert isinstance(result, ChatCompletionResponse)
        assert len(result.choices) > 0
        assert result.choices[0].message.role == "assistant"


# =============================================================================
# Test: Output Comparison (AC-14.3)
# =============================================================================


class TestDebateModeComparison:
    """Tests for output comparison and agreement calculation."""

    async def test_debate_mode_calculates_agreement_percentage(
        self,
        mock_provider_a: MagicMock,
        mock_provider_b: MagicMock,
        mock_reconciler: MagicMock,
        sample_request: ChatCompletionRequest,
    ) -> None:
        """Should calculate agreement percentage between outputs."""
        from src.orchestration.modes.debate import DebateMode

        mode = DebateMode(
            participant_a=mock_provider_a,
            participant_b=mock_provider_b,
            reconciler=mock_reconciler,
        )

        result = await mode.execute(sample_request)

        # agreement_score should reflect agreement
        assert result.orchestration is not None
        assert result.orchestration.agreement_score is not None
        assert 0.0 <= result.orchestration.agreement_score <= 1.0

    async def test_debate_mode_high_agreement_for_similar_responses(
        self,
        mock_provider_a: MagicMock,
        mock_provider_b: MagicMock,
        mock_reconciler: MagicMock,
        sample_request: ChatCompletionRequest,
    ) -> None:
        """Similar responses should have high agreement."""
        from src.orchestration.modes.debate import DebateMode

        # Create identical responses
        identical_response = ChatCompletionResponse(
            id="identical",
            created=0,
            model="test",
            choices=[
                Choice(
                    index=0,
                    message=ChoiceMessage(
                        role="assistant",
                        content="The capital of France is Paris.",
                    ),
                    finish_reason="stop",
                )
            ],
            usage=Usage(prompt_tokens=10, completion_tokens=10, total_tokens=20),
        )

        mock_provider_a.generate = AsyncMock(return_value=identical_response)
        mock_provider_b.generate = AsyncMock(return_value=identical_response)

        mode = DebateMode(
            participant_a=mock_provider_a,
            participant_b=mock_provider_b,
            reconciler=mock_reconciler,
        )

        result = await mode.execute(sample_request)

        # Identical responses should have 100% agreement
        assert result.orchestration is not None
        assert result.orchestration.agreement_score == 1.0

    async def test_debate_mode_low_agreement_for_different_responses(
        self,
        mock_provider_a: MagicMock,
        mock_provider_b: MagicMock,
        mock_reconciler: MagicMock,
        sample_request: ChatCompletionRequest,
    ) -> None:
        """Different responses should have lower agreement."""
        from src.orchestration.modes.debate import DebateMode

        response_a = ChatCompletionResponse(
            id="a",
            created=0,
            model="a",
            choices=[
                Choice(
                    index=0,
                    message=ChoiceMessage(
                        role="assistant",
                        content="The answer is definitely yes.",
                    ),
                    finish_reason="stop",
                )
            ],
            usage=Usage(prompt_tokens=10, completion_tokens=10, total_tokens=20),
        )

        response_b = ChatCompletionResponse(
            id="b",
            created=0,
            model="b",
            choices=[
                Choice(
                    index=0,
                    message=ChoiceMessage(
                        role="assistant",
                        content="No, absolutely not.",
                    ),
                    finish_reason="stop",
                )
            ],
            usage=Usage(prompt_tokens=10, completion_tokens=10, total_tokens=20),
        )

        mock_provider_a.generate = AsyncMock(return_value=response_a)
        mock_provider_b.generate = AsyncMock(return_value=response_b)

        mode = DebateMode(
            participant_a=mock_provider_a,
            participant_b=mock_provider_b,
            reconciler=mock_reconciler,
        )

        result = await mode.execute(sample_request)

        # Different responses should have lower agreement
        assert result.orchestration is not None
        assert result.orchestration.agreement_score < 0.5

    def test_calculate_agreement_identical_strings(self) -> None:
        """Identical strings should have 100% agreement."""
        from src.orchestration.modes.debate import calculate_agreement

        text_a = "The capital of France is Paris."
        text_b = "The capital of France is Paris."

        agreement = calculate_agreement(text_a, text_b)
        assert agreement == 1.0

    def test_calculate_agreement_completely_different(self) -> None:
        """Completely different strings should have low agreement."""
        from src.orchestration.modes.debate import calculate_agreement

        text_a = "yes yes yes yes yes"
        text_b = "no no no no no"

        agreement = calculate_agreement(text_a, text_b)
        assert agreement < 0.2

    def test_calculate_agreement_partial_overlap(self) -> None:
        """Partially overlapping strings should have medium agreement."""
        from src.orchestration.modes.debate import calculate_agreement

        text_a = "Paris is the capital of France."
        text_b = "Paris is the largest city in France."

        agreement = calculate_agreement(text_a, text_b)
        assert 0.3 < agreement < 0.9

    def test_calculate_agreement_case_insensitive(self) -> None:
        """Agreement should be case insensitive."""
        from src.orchestration.modes.debate import calculate_agreement

        text_a = "PARIS is the CAPITAL"
        text_b = "paris is the capital"

        agreement = calculate_agreement(text_a, text_b)
        assert agreement == 1.0

    def test_calculate_agreement_empty_strings(self) -> None:
        """Empty strings should have agreement of 1.0 (vacuous truth)."""
        from src.orchestration.modes.debate import calculate_agreement

        agreement = calculate_agreement("", "")
        assert agreement == 1.0

    def test_calculate_agreement_one_empty(self) -> None:
        """One empty string should have 0 agreement."""
        from src.orchestration.modes.debate import calculate_agreement

        agreement = calculate_agreement("Some text", "")
        assert agreement == 0.0


# =============================================================================
# Test: Reconciliation (AC-14.4)
# =============================================================================


class TestDebateModeReconciliation:
    """Tests for reconciler synthesis."""

    async def test_debate_mode_calls_reconciler(
        self,
        mock_provider_a: MagicMock,
        mock_provider_b: MagicMock,
        mock_reconciler: MagicMock,
        sample_request: ChatCompletionRequest,
    ) -> None:
        """Reconciler should be called after parallel generation."""
        from src.orchestration.modes.debate import DebateMode

        mode = DebateMode(
            participant_a=mock_provider_a,
            participant_b=mock_provider_b,
            reconciler=mock_reconciler,
        )

        await mode.execute(sample_request)

        mock_reconciler.generate.assert_called_once()

    async def test_debate_mode_reconciler_receives_both_outputs(
        self,
        mock_provider_a: MagicMock,
        mock_provider_b: MagicMock,
        mock_reconciler: MagicMock,
        sample_request: ChatCompletionRequest,
    ) -> None:
        """Reconciler should receive both participant outputs."""
        from src.orchestration.modes.debate import DebateMode

        mode = DebateMode(
            participant_a=mock_provider_a,
            participant_b=mock_provider_b,
            reconciler=mock_reconciler,
        )

        await mode.execute(sample_request)

        # Check that reconciler received both outputs in the prompt
        call_args = mock_reconciler.generate.call_args
        request: ChatCompletionRequest = call_args[0][0]

        # Find the message containing both outputs
        messages_content = " ".join(m.content or "" for m in request.messages)
        assert "Paris has been the capital since 987 AD" in messages_content
        assert "largest city in France" in messages_content

    async def test_debate_mode_returns_reconciled_content(
        self,
        mock_provider_a: MagicMock,
        mock_provider_b: MagicMock,
        mock_reconciler: MagicMock,
        sample_request: ChatCompletionRequest,
    ) -> None:
        """Final response should be from reconciler."""
        from src.orchestration.modes.debate import DebateMode

        mode = DebateMode(
            participant_a=mock_provider_a,
            participant_b=mock_provider_b,
            reconciler=mock_reconciler,
        )

        result = await mode.execute(sample_request)

        # Result should be from reconciler
        expected = "The capital of France is Paris. Paris has been the capital since 987 AD and is also the largest city in France."
        assert result.choices[0].message.content == expected

    async def test_debate_mode_reconciler_can_be_same_as_participant(
        self,
        mock_provider_a: MagicMock,
        mock_provider_b: MagicMock,
        sample_request: ChatCompletionRequest,
    ) -> None:
        """Reconciler can be the same provider as a participant."""
        from src.orchestration.modes.debate import DebateMode

        # Use provider_a as both participant and reconciler (as per ARCHITECTURE.md)
        mock_provider_a.generate = AsyncMock(
            side_effect=[
                ChatCompletionResponse(
                    id="a",
                    created=0,
                    model="a",
                    choices=[
                        Choice(
                            index=0,
                            message=ChoiceMessage(
                                role="assistant", content="Response A"
                            ),
                            finish_reason="stop",
                        )
                    ],
                    usage=Usage(prompt_tokens=10, completion_tokens=10, total_tokens=20),
                ),
                ChatCompletionResponse(
                    id="reconciled",
                    created=0,
                    model="a",
                    choices=[
                        Choice(
                            index=0,
                            message=ChoiceMessage(
                                role="assistant", content="Synthesized"
                            ),
                            finish_reason="stop",
                        )
                    ],
                    usage=Usage(prompt_tokens=10, completion_tokens=10, total_tokens=20),
                ),
            ]
        )

        mode = DebateMode(
            participant_a=mock_provider_a,
            participant_b=mock_provider_b,
            reconciler=mock_provider_a,
        )

        result = await mode.execute(sample_request)

        # Should have called provider_a twice (once for generation, once for reconciliation)
        assert mock_provider_a.generate.call_count == 2
        assert result.choices[0].message.content == "Synthesized"


# =============================================================================
# Test: Metadata (AC-14.4)
# =============================================================================


class TestDebateModeMetadata:
    """Tests for orchestration metadata."""

    async def test_debate_mode_sets_mode_to_debate(
        self,
        mock_provider_a: MagicMock,
        mock_provider_b: MagicMock,
        mock_reconciler: MagicMock,
        sample_request: ChatCompletionRequest,
    ) -> None:
        """Metadata should indicate debate mode."""
        from src.orchestration.modes.debate import DebateMode

        mode = DebateMode(
            participant_a=mock_provider_a,
            participant_b=mock_provider_b,
            reconciler=mock_reconciler,
        )

        result = await mode.execute(sample_request)

        assert result.orchestration is not None
        assert result.orchestration.mode == "debate"

    async def test_debate_mode_reports_all_models_used(
        self,
        mock_provider_a: MagicMock,
        mock_provider_b: MagicMock,
        mock_reconciler: MagicMock,
        sample_request: ChatCompletionRequest,
    ) -> None:
        """Metadata should list all models used."""
        from src.orchestration.modes.debate import DebateMode

        mode = DebateMode(
            participant_a=mock_provider_a,
            participant_b=mock_provider_b,
            reconciler=mock_reconciler,
        )

        result = await mode.execute(sample_request)

        assert result.orchestration is not None
        models = result.orchestration.models_used
        assert "model-a" in models
        assert "model-b" in models
        assert "reconciler-model" in models

    async def test_debate_mode_reports_inference_time(
        self,
        mock_provider_a: MagicMock,
        mock_provider_b: MagicMock,
        mock_reconciler: MagicMock,
        sample_request: ChatCompletionRequest,
    ) -> None:
        """Metadata should include inference time."""
        from src.orchestration.modes.debate import DebateMode

        mode = DebateMode(
            participant_a=mock_provider_a,
            participant_b=mock_provider_b,
            reconciler=mock_reconciler,
        )

        result = await mode.execute(sample_request)

        assert result.orchestration is not None
        assert result.orchestration.total_inference_time_ms >= 0

    async def test_debate_mode_aggregates_usage_stats(
        self,
        mock_provider_a: MagicMock,
        mock_provider_b: MagicMock,
        mock_reconciler: MagicMock,
        sample_request: ChatCompletionRequest,
    ) -> None:
        """Usage should aggregate across all calls."""
        from src.orchestration.modes.debate import DebateMode

        mode = DebateMode(
            participant_a=mock_provider_a,
            participant_b=mock_provider_b,
            reconciler=mock_reconciler,
        )

        result = await mode.execute(sample_request)

        # Total usage should be sum of A (30) + B (28) + reconciler (80)
        assert result.usage is not None
        assert result.usage.total_tokens == 30 + 28 + 80


# =============================================================================
# Test: Properties
# =============================================================================


class TestDebateModeProperties:
    """Tests for DebateMode properties."""

    def test_debate_mode_participant_a_property(
        self,
        mock_provider_a: MagicMock,
        mock_provider_b: MagicMock,
        mock_reconciler: MagicMock,
    ) -> None:
        """Should expose participant_a provider."""
        from src.orchestration.modes.debate import DebateMode

        mode = DebateMode(
            participant_a=mock_provider_a,
            participant_b=mock_provider_b,
            reconciler=mock_reconciler,
        )

        assert mode.participant_a is mock_provider_a

    def test_debate_mode_participant_b_property(
        self,
        mock_provider_a: MagicMock,
        mock_provider_b: MagicMock,
        mock_reconciler: MagicMock,
    ) -> None:
        """Should expose participant_b provider."""
        from src.orchestration.modes.debate import DebateMode

        mode = DebateMode(
            participant_a=mock_provider_a,
            participant_b=mock_provider_b,
            reconciler=mock_reconciler,
        )

        assert mode.participant_b is mock_provider_b

    def test_debate_mode_reconciler_property(
        self,
        mock_provider_a: MagicMock,
        mock_provider_b: MagicMock,
        mock_reconciler: MagicMock,
    ) -> None:
        """Should expose reconciler provider."""
        from src.orchestration.modes.debate import DebateMode

        mode = DebateMode(
            participant_a=mock_provider_a,
            participant_b=mock_provider_b,
            reconciler=mock_reconciler,
        )

        assert mode.reconciler is mock_reconciler

    def test_debate_mode_model_ids(
        self,
        mock_provider_a: MagicMock,
        mock_provider_b: MagicMock,
        mock_reconciler: MagicMock,
    ) -> None:
        """Should expose model IDs for all participants."""
        from src.orchestration.modes.debate import DebateMode

        mode = DebateMode(
            participant_a=mock_provider_a,
            participant_b=mock_provider_b,
            reconciler=mock_reconciler,
        )

        assert mode.participant_a_model_id == "model-a"
        assert mode.participant_b_model_id == "model-b"
        assert mode.reconciler_model_id == "reconciler-model"


# =============================================================================
# Test: Edge Cases
# =============================================================================


class TestDebateModeEdgeCases:
    """Tests for edge cases in DebateMode."""

    async def test_debate_mode_handles_empty_response(
        self,
        mock_provider_a: MagicMock,
        mock_provider_b: MagicMock,
        mock_reconciler: MagicMock,
        sample_request: ChatCompletionRequest,
    ) -> None:
        """Should handle empty response from a participant."""
        from src.orchestration.modes.debate import DebateMode

        empty_response = ChatCompletionResponse(
            id="empty",
            created=0,
            model="a",
            choices=[
                Choice(
                    index=0,
                    message=ChoiceMessage(role="assistant", content=""),
                    finish_reason="stop",
                )
            ],
            usage=Usage(prompt_tokens=10, completion_tokens=0, total_tokens=10),
        )

        mock_provider_a.generate = AsyncMock(return_value=empty_response)

        mode = DebateMode(
            participant_a=mock_provider_a,
            participant_b=mock_provider_b,
            reconciler=mock_reconciler,
        )

        # Should still complete without error
        result = await mode.execute(sample_request)
        assert isinstance(result, ChatCompletionResponse)

    async def test_debate_mode_two_providers_same_instance(
        self,
        mock_provider_a: MagicMock,
        mock_reconciler: MagicMock,
        sample_request: ChatCompletionRequest,
    ) -> None:
        """Should work with same provider instance for both participants."""
        from src.orchestration.modes.debate import DebateMode

        # Make provider return different responses on each call
        mock_provider_a.generate = AsyncMock(
            side_effect=[
                ChatCompletionResponse(
                    id="a1",
                    created=0,
                    model="a",
                    choices=[
                        Choice(
                            index=0,
                            message=ChoiceMessage(role="assistant", content="First"),
                            finish_reason="stop",
                        )
                    ],
                    usage=Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
                ),
                ChatCompletionResponse(
                    id="a2",
                    created=0,
                    model="a",
                    choices=[
                        Choice(
                            index=0,
                            message=ChoiceMessage(role="assistant", content="Second"),
                            finish_reason="stop",
                        )
                    ],
                    usage=Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
                ),
            ]
        )

        mode = DebateMode(
            participant_a=mock_provider_a,
            participant_b=mock_provider_a,  # Same provider
            reconciler=mock_reconciler,
        )

        # This is an unusual case but should work
        result = await mode.execute(sample_request)
        assert isinstance(result, ChatCompletionResponse)

    async def test_debate_mode_preserves_model_field(
        self,
        mock_provider_a: MagicMock,
        mock_provider_b: MagicMock,
        mock_reconciler: MagicMock,
        sample_request: ChatCompletionRequest,
    ) -> None:
        """Response should preserve model field from reconciler."""
        from src.orchestration.modes.debate import DebateMode

        mode = DebateMode(
            participant_a=mock_provider_a,
            participant_b=mock_provider_b,
            reconciler=mock_reconciler,
        )

        result = await mode.execute(sample_request)

        # Model should be from reconciler
        assert result.model == "reconciler-model"
