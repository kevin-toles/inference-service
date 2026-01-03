"""Unit tests for EnsembleMode orchestration.

Tests EnsembleMode functionality including:
- Parallel generation from all models (AC-15.1)
- Consensus score calculation (AC-15.2)
- Synthesis from agreed points (AC-15.3)
- Disagreement flagging (AC-15.4)

Reference: WBS-INF15
"""

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.models.requests import ChatCompletionRequest, Message
from src.models.responses import (
    ChatCompletionResponse,
    Choice,
    ChoiceMessage,
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


def create_mock_response(
    model_id: str, content: str, prompt_tokens: int = 10, completion_tokens: int = 20
) -> ChatCompletionResponse:
    """Create a mock response with given parameters."""
    return ChatCompletionResponse(
        id=f"chatcmpl-{model_id}",
        created=1234567890,
        model=model_id,
        choices=[
            Choice(
                index=0,
                message=ChoiceMessage(role="assistant", content=content),
                finish_reason="stop",
            )
        ],
        usage=Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        ),
    )


def create_mock_provider(model_id: str, response: ChatCompletionResponse) -> MagicMock:
    """Create a mock provider with given model ID and response."""
    provider = MagicMock()
    provider.generate = AsyncMock(return_value=response)
    provider.model_info = MagicMock()
    provider.model_info.model_id = model_id
    return provider


@pytest.fixture
def mock_providers_agreeing() -> list[MagicMock]:
    """Create 3 mock providers that agree on the answer."""
    responses = [
        create_mock_response("model-a", "The capital of France is Paris."),
        create_mock_response("model-b", "Paris is the capital of France."),
        create_mock_response("model-c", "France's capital city is Paris."),
    ]
    return [
        create_mock_provider("model-a", responses[0]),
        create_mock_provider("model-b", responses[1]),
        create_mock_provider("model-c", responses[2]),
    ]


@pytest.fixture
def mock_providers_disagreeing() -> list[MagicMock]:
    """Create 3 mock providers with disagreement."""
    responses = [
        create_mock_response("model-a", "The capital of France is Paris."),
        create_mock_response("model-b", "Paris is the capital of France."),
        create_mock_response("model-c", "The capital of France is Lyon."),  # Wrong!
    ]
    return [
        create_mock_provider("model-a", responses[0]),
        create_mock_provider("model-b", responses[1]),
        create_mock_provider("model-c", responses[2]),
    ]


@pytest.fixture
def mock_synthesizer() -> MagicMock:
    """Create a mock synthesizer provider."""
    response = create_mock_response(
        "synthesizer",
        "The capital of France is Paris. This is confirmed by consensus.",
        prompt_tokens=50,
        completion_tokens=30,
    )
    return create_mock_provider("synthesizer", response)


# =============================================================================
# Test: Parallel Generation (AC-15.1)
# =============================================================================


class TestEnsembleModeParallelGeneration:
    """Tests for parallel generation in EnsembleMode."""

    async def test_ensemble_mode_calls_all_providers(
        self,
        mock_providers_agreeing: list[MagicMock],
        mock_synthesizer: MagicMock,
        sample_request: ChatCompletionRequest,
    ) -> None:
        """All providers should be called during ensemble."""
        from src.orchestration.modes.ensemble import EnsembleMode

        mode = EnsembleMode(
            participants=mock_providers_agreeing,
            synthesizer=mock_synthesizer,
        )

        await mode.execute(sample_request)

        for provider in mock_providers_agreeing:
            provider.generate.assert_called_once()

    async def test_ensemble_mode_uses_asyncio_gather_for_parallel(
        self,
        mock_synthesizer: MagicMock,
        sample_request: ChatCompletionRequest,
    ) -> None:
        """Providers should be called in parallel using asyncio.gather."""
        from src.orchestration.modes.ensemble import EnsembleMode

        # Track call times
        call_times: list[tuple[str, float]] = []

        async def create_delayed_generate(model_id: str) -> ChatCompletionResponse:
            async def delayed_generate(
                request: ChatCompletionRequest,
            ) -> ChatCompletionResponse:
                call_times.append((f"{model_id}_start", time.perf_counter()))
                await asyncio.sleep(0.03)  # 30ms delay
                call_times.append((f"{model_id}_end", time.perf_counter()))
                return create_mock_response(model_id, f"Response from {model_id}")

            await asyncio.sleep(0)  # Satisfy async requirement
            return delayed_generate

        providers = []
        for i, model_id in enumerate(["model-a", "model-b", "model-c"]):
            provider = MagicMock()
            provider.generate = await create_delayed_generate(model_id)
            provider.model_info = MagicMock()
            provider.model_info.model_id = model_id
            providers.append(provider)

        mode = EnsembleMode(participants=providers, synthesizer=mock_synthesizer)

        start = time.perf_counter()
        await mode.execute(sample_request)
        elapsed = time.perf_counter() - start

        # Parallel execution: 3 x 30ms should be ~30-50ms, not 90ms
        # Allow margin for overhead but should be < 100ms
        assert elapsed < 0.1, f"Expected parallel execution, but took {elapsed}s"

        # Verify overlapping execution
        start_times = [t for name, t in call_times if "start" in name]
        end_times = [t for name, t in call_times if "end" in name]
        assert min(end_times) > max(start_times), "Expected overlapping execution"

    async def test_ensemble_mode_minimum_two_providers(
        self,
        mock_synthesizer: MagicMock,
        sample_request: ChatCompletionRequest,
    ) -> None:
        """Ensemble should work with minimum 2 providers."""
        from src.orchestration.modes.ensemble import EnsembleMode

        providers = [
            create_mock_provider("model-a", create_mock_response("model-a", "Answer A")),
            create_mock_provider("model-b", create_mock_response("model-b", "Answer B")),
        ]

        mode = EnsembleMode(participants=providers, synthesizer=mock_synthesizer)
        result = await mode.execute(sample_request)

        assert isinstance(result, ChatCompletionResponse)

    async def test_ensemble_mode_handles_single_provider_error(
        self,
        mock_providers_agreeing: list[MagicMock],
        mock_synthesizer: MagicMock,
        sample_request: ChatCompletionRequest,
    ) -> None:
        """Should propagate error if a provider fails."""
        from src.orchestration.modes.ensemble import EnsembleMode

        mock_providers_agreeing[1].generate = AsyncMock(
            side_effect=Exception("Model B failed")
        )

        mode = EnsembleMode(
            participants=mock_providers_agreeing,
            synthesizer=mock_synthesizer,
        )

        with pytest.raises(Exception, match="Model B failed"):
            await mode.execute(sample_request)

    async def test_ensemble_mode_returns_chat_completion_response(
        self,
        mock_providers_agreeing: list[MagicMock],
        mock_synthesizer: MagicMock,
        sample_request: ChatCompletionRequest,
    ) -> None:
        """Should return a valid ChatCompletionResponse."""
        from src.orchestration.modes.ensemble import EnsembleMode

        mode = EnsembleMode(
            participants=mock_providers_agreeing,
            synthesizer=mock_synthesizer,
        )

        result = await mode.execute(sample_request)

        assert isinstance(result, ChatCompletionResponse)
        assert len(result.choices) > 0
        assert result.choices[0].message.role == "assistant"


# =============================================================================
# Test: Consensus Calculation (AC-15.2)
# =============================================================================


class TestEnsembleModeConsensus:
    """Tests for consensus score calculation."""

    async def test_ensemble_mode_calculates_consensus_score(
        self,
        mock_providers_agreeing: list[MagicMock],
        mock_synthesizer: MagicMock,
        sample_request: ChatCompletionRequest,
    ) -> None:
        """Should calculate consensus score between all outputs."""
        from src.orchestration.modes.ensemble import EnsembleMode

        mode = EnsembleMode(
            participants=mock_providers_agreeing,
            synthesizer=mock_synthesizer,
        )

        result = await mode.execute(sample_request)

        assert result.orchestration is not None
        assert result.orchestration.agreement_score is not None
        assert 0.0 <= result.orchestration.agreement_score <= 1.0

    async def test_ensemble_mode_high_consensus_for_similar_responses(
        self,
        mock_synthesizer: MagicMock,
        sample_request: ChatCompletionRequest,
    ) -> None:
        """Similar responses should have high consensus score."""
        from src.orchestration.modes.ensemble import EnsembleMode

        # All providers return nearly identical content
        identical_content = "The capital of France is Paris."
        providers = [
            create_mock_provider(
                f"model-{i}", create_mock_response(f"model-{i}", identical_content)
            )
            for i in range(3)
        ]

        mode = EnsembleMode(participants=providers, synthesizer=mock_synthesizer)
        result = await mode.execute(sample_request)

        assert result.orchestration is not None
        assert result.orchestration.agreement_score == pytest.approx(1.0)

    async def test_ensemble_mode_lower_consensus_for_disagreement(
        self,
        mock_providers_disagreeing: list[MagicMock],
        mock_synthesizer: MagicMock,
        sample_request: ChatCompletionRequest,
    ) -> None:
        """Disagreeing responses should have lower consensus score."""
        from src.orchestration.modes.ensemble import EnsembleMode

        mode = EnsembleMode(
            participants=mock_providers_disagreeing,
            synthesizer=mock_synthesizer,
        )

        result = await mode.execute(sample_request)

        assert result.orchestration is not None
        # Not perfect consensus due to one wrong answer
        assert result.orchestration.agreement_score < 1.0

    def test_calculate_pairwise_consensus_identical(self) -> None:
        """Identical responses should have 100% pairwise consensus."""
        from src.orchestration.modes.ensemble import calculate_pairwise_consensus

        responses = ["Paris is the capital.", "Paris is the capital.", "Paris is the capital."]
        consensus = calculate_pairwise_consensus(responses)
        assert consensus == pytest.approx(1.0)

    def test_calculate_pairwise_consensus_partial(self) -> None:
        """Partially agreeing responses should have medium consensus."""
        from src.orchestration.modes.ensemble import calculate_pairwise_consensus

        responses = [
            "Paris is the capital of France.",
            "Paris is the capital of France.",
            "Lyon is the capital of France.",  # Disagrees
        ]
        consensus = calculate_pairwise_consensus(responses)
        # Two pairs agree (A-B), two pairs partially agree (A-C, B-C)
        assert 0.3 < consensus < 1.0

    def test_calculate_pairwise_consensus_completely_different(self) -> None:
        """Completely different responses should have low consensus."""
        from src.orchestration.modes.ensemble import calculate_pairwise_consensus

        responses = ["yes yes yes", "no no no", "maybe maybe maybe"]
        consensus = calculate_pairwise_consensus(responses)
        assert consensus < 0.3

    def test_calculate_pairwise_consensus_two_responses(self) -> None:
        """Should work with just two responses."""
        from src.orchestration.modes.ensemble import calculate_pairwise_consensus

        responses = ["The answer is Paris.", "The answer is Paris."]
        consensus = calculate_pairwise_consensus(responses)
        assert consensus == pytest.approx(1.0)

    def test_calculate_pairwise_consensus_empty_list(self) -> None:
        """Empty list should return 1.0 (vacuous truth)."""
        from src.orchestration.modes.ensemble import calculate_pairwise_consensus

        consensus = calculate_pairwise_consensus([])
        assert consensus == pytest.approx(1.0)

    def test_calculate_pairwise_consensus_single_response(self) -> None:
        """Single response should return 1.0."""
        from src.orchestration.modes.ensemble import calculate_pairwise_consensus

        consensus = calculate_pairwise_consensus(["Only one response"])
        assert consensus == pytest.approx(1.0)


# =============================================================================
# Test: Synthesis (AC-15.3)
# =============================================================================


class TestEnsembleModeSynthesis:
    """Tests for synthesizing from agreed points."""

    async def test_ensemble_mode_calls_synthesizer(
        self,
        mock_providers_agreeing: list[MagicMock],
        mock_synthesizer: MagicMock,
        sample_request: ChatCompletionRequest,
    ) -> None:
        """Synthesizer should be called after parallel generation."""
        from src.orchestration.modes.ensemble import EnsembleMode

        mode = EnsembleMode(
            participants=mock_providers_agreeing,
            synthesizer=mock_synthesizer,
        )

        await mode.execute(sample_request)

        mock_synthesizer.generate.assert_called_once()

    async def test_ensemble_mode_synthesizer_receives_all_outputs(
        self,
        mock_providers_agreeing: list[MagicMock],
        mock_synthesizer: MagicMock,
        sample_request: ChatCompletionRequest,
    ) -> None:
        """Synthesizer should receive all participant outputs."""
        from src.orchestration.modes.ensemble import EnsembleMode

        mode = EnsembleMode(
            participants=mock_providers_agreeing,
            synthesizer=mock_synthesizer,
        )

        await mode.execute(sample_request)

        # Check that synthesizer received all outputs in the prompt
        call_args = mock_synthesizer.generate.call_args
        request: ChatCompletionRequest = call_args[0][0]

        messages_content = " ".join(m.content or "" for m in request.messages)
        # All three responses should be included
        assert "Paris" in messages_content
        assert "capital" in messages_content

    async def test_ensemble_mode_returns_synthesized_content(
        self,
        mock_providers_agreeing: list[MagicMock],
        mock_synthesizer: MagicMock,
        sample_request: ChatCompletionRequest,
    ) -> None:
        """Final response should be from synthesizer."""
        from src.orchestration.modes.ensemble import EnsembleMode

        mode = EnsembleMode(
            participants=mock_providers_agreeing,
            synthesizer=mock_synthesizer,
        )

        result = await mode.execute(sample_request)

        expected = "The capital of France is Paris. This is confirmed by consensus."
        assert result.choices[0].message.content == expected

    async def test_ensemble_mode_synthesizer_can_be_participant(
        self,
        sample_request: ChatCompletionRequest,
    ) -> None:
        """Synthesizer can be one of the participants."""
        from src.orchestration.modes.ensemble import EnsembleMode

        # Model A is both participant and synthesizer
        responses = [
            create_mock_response("model-a", "Answer from A"),
            create_mock_response("model-a", "Synthesized answer"),
        ]

        provider_a = MagicMock()
        provider_a.generate = AsyncMock(side_effect=responses)
        provider_a.model_info = MagicMock()
        provider_a.model_info.model_id = "model-a"

        provider_b = create_mock_provider(
            "model-b", create_mock_response("model-b", "Answer from B")
        )

        mode = EnsembleMode(
            participants=[provider_a, provider_b],
            synthesizer=provider_a,
        )

        result = await mode.execute(sample_request)

        # Should have called provider_a twice (participant + synthesizer)
        assert provider_a.generate.call_count == 2
        assert result.choices[0].message.content == "Synthesized answer"


# =============================================================================
# Test: Disagreement Flagging (AC-15.4)
# =============================================================================


class TestEnsembleModeDisagreement:
    """Tests for flagging disagreements."""

    async def test_ensemble_mode_flags_low_consensus(
        self,
        mock_synthesizer: MagicMock,
        sample_request: ChatCompletionRequest,
    ) -> None:
        """Should flag when consensus is below threshold."""
        from src.orchestration.modes.ensemble import EnsembleMode

        # Create highly disagreeing providers
        providers = [
            create_mock_provider(
                "model-a", create_mock_response("model-a", "yes yes yes yes")
            ),
            create_mock_provider(
                "model-b", create_mock_response("model-b", "no no no no")
            ),
            create_mock_provider(
                "model-c", create_mock_response("model-c", "maybe maybe maybe")
            ),
        ]

        mode = EnsembleMode(
            participants=providers,
            synthesizer=mock_synthesizer,
            min_agreement=0.7,  # Threshold
        )

        result = await mode.execute(sample_request)

        # Consensus should be low
        assert result.orchestration is not None
        assert result.orchestration.agreement_score < 0.7

    def test_extract_disagreement_points_simple(self) -> None:
        """Should identify disagreement points between responses."""
        from src.orchestration.modes.ensemble import extract_disagreement_points

        responses = [
            "The capital is Paris.",
            "The capital is Paris.",
            "The capital is Lyon.",
        ]

        disagreements = extract_disagreement_points(responses)

        # Should have some disagreement noted
        assert len(disagreements) > 0
        # Lyon appears in only one response
        assert any("Lyon" in d or "lyon" in d.lower() for d in disagreements)

    def test_extract_disagreement_points_no_disagreement(self) -> None:
        """Should return empty list when all agree."""
        from src.orchestration.modes.ensemble import extract_disagreement_points

        responses = [
            "Paris is the capital.",
            "Paris is the capital.",
            "Paris is the capital.",
        ]

        disagreements = extract_disagreement_points(responses)
        assert len(disagreements) == 0

    def test_extract_disagreement_points_majority_vote(self) -> None:
        """Should identify minority opinions as disagreements."""
        from src.orchestration.modes.ensemble import extract_disagreement_points

        responses = [
            "The answer is 42.",
            "The answer is 42.",
            "The answer is 42.",
            "The answer is 7.",  # Minority
        ]

        disagreements = extract_disagreement_points(responses)
        # The minority opinion should be flagged
        assert any("7" in d for d in disagreements)


# =============================================================================
# Test: Metadata
# =============================================================================


class TestEnsembleModeMetadata:
    """Tests for orchestration metadata."""

    async def test_ensemble_mode_sets_mode_to_ensemble(
        self,
        mock_providers_agreeing: list[MagicMock],
        mock_synthesizer: MagicMock,
        sample_request: ChatCompletionRequest,
    ) -> None:
        """Metadata should indicate ensemble mode."""
        from src.orchestration.modes.ensemble import EnsembleMode

        mode = EnsembleMode(
            participants=mock_providers_agreeing,
            synthesizer=mock_synthesizer,
        )

        result = await mode.execute(sample_request)

        assert result.orchestration is not None
        assert result.orchestration.mode == "ensemble"

    async def test_ensemble_mode_reports_all_models_used(
        self,
        mock_providers_agreeing: list[MagicMock],
        mock_synthesizer: MagicMock,
        sample_request: ChatCompletionRequest,
    ) -> None:
        """Metadata should list all models used."""
        from src.orchestration.modes.ensemble import EnsembleMode

        mode = EnsembleMode(
            participants=mock_providers_agreeing,
            synthesizer=mock_synthesizer,
        )

        result = await mode.execute(sample_request)

        assert result.orchestration is not None
        models = result.orchestration.models_used
        assert "model-a" in models
        assert "model-b" in models
        assert "model-c" in models
        assert "synthesizer" in models

    async def test_ensemble_mode_reports_inference_time(
        self,
        mock_providers_agreeing: list[MagicMock],
        mock_synthesizer: MagicMock,
        sample_request: ChatCompletionRequest,
    ) -> None:
        """Metadata should include inference time."""
        from src.orchestration.modes.ensemble import EnsembleMode

        mode = EnsembleMode(
            participants=mock_providers_agreeing,
            synthesizer=mock_synthesizer,
        )

        result = await mode.execute(sample_request)

        assert result.orchestration is not None
        assert result.orchestration.total_inference_time_ms >= 0

    async def test_ensemble_mode_aggregates_usage_stats(
        self,
        mock_providers_agreeing: list[MagicMock],
        mock_synthesizer: MagicMock,
        sample_request: ChatCompletionRequest,
    ) -> None:
        """Usage should aggregate across all calls."""
        from src.orchestration.modes.ensemble import EnsembleMode

        mode = EnsembleMode(
            participants=mock_providers_agreeing,
            synthesizer=mock_synthesizer,
        )

        result = await mode.execute(sample_request)

        # Total usage should be sum of all providers + synthesizer
        # 3 participants: 30 each = 90, synthesizer: 80
        assert result.usage is not None
        assert result.usage.total_tokens == 90 + 80


# =============================================================================
# Test: Properties
# =============================================================================


class TestEnsembleModeProperties:
    """Tests for EnsembleMode properties."""

    def test_ensemble_mode_participants_property(
        self,
        mock_providers_agreeing: list[MagicMock],
        mock_synthesizer: MagicMock,
    ) -> None:
        """Should expose participants list."""
        from src.orchestration.modes.ensemble import EnsembleMode

        mode = EnsembleMode(
            participants=mock_providers_agreeing,
            synthesizer=mock_synthesizer,
        )

        assert mode.participants == mock_providers_agreeing

    def test_ensemble_mode_synthesizer_property(
        self,
        mock_providers_agreeing: list[MagicMock],
        mock_synthesizer: MagicMock,
    ) -> None:
        """Should expose synthesizer provider."""
        from src.orchestration.modes.ensemble import EnsembleMode

        mode = EnsembleMode(
            participants=mock_providers_agreeing,
            synthesizer=mock_synthesizer,
        )

        assert mode.synthesizer is mock_synthesizer

    def test_ensemble_mode_min_agreement_property(
        self,
        mock_providers_agreeing: list[MagicMock],
        mock_synthesizer: MagicMock,
    ) -> None:
        """Should expose min_agreement threshold."""
        from src.orchestration.modes.ensemble import EnsembleMode

        mode = EnsembleMode(
            participants=mock_providers_agreeing,
            synthesizer=mock_synthesizer,
            min_agreement=0.8,
        )

        assert mode.min_agreement == pytest.approx(0.8)

    def test_ensemble_mode_default_min_agreement(
        self,
        mock_providers_agreeing: list[MagicMock],
        mock_synthesizer: MagicMock,
    ) -> None:
        """Default min_agreement should be 0.7."""
        from src.orchestration.modes.ensemble import EnsembleMode

        mode = EnsembleMode(
            participants=mock_providers_agreeing,
            synthesizer=mock_synthesizer,
        )

        assert mode.min_agreement == pytest.approx(0.7)

    def test_ensemble_mode_participant_model_ids(
        self,
        mock_providers_agreeing: list[MagicMock],
        mock_synthesizer: MagicMock,
    ) -> None:
        """Should expose participant model IDs."""
        from src.orchestration.modes.ensemble import EnsembleMode

        mode = EnsembleMode(
            participants=mock_providers_agreeing,
            synthesizer=mock_synthesizer,
        )

        ids = mode.participant_model_ids
        assert "model-a" in ids
        assert "model-b" in ids
        assert "model-c" in ids


# =============================================================================
# Test: Edge Cases
# =============================================================================


class TestEnsembleModeEdgeCases:
    """Tests for edge cases in EnsembleMode."""

    async def test_ensemble_mode_handles_empty_response(
        self,
        mock_providers_agreeing: list[MagicMock],
        mock_synthesizer: MagicMock,
        sample_request: ChatCompletionRequest,
    ) -> None:
        """Should handle empty response from a participant."""
        from src.orchestration.modes.ensemble import EnsembleMode

        # One provider returns empty content
        mock_providers_agreeing[1].generate = AsyncMock(
            return_value=create_mock_response("model-b", "")
        )

        mode = EnsembleMode(
            participants=mock_providers_agreeing,
            synthesizer=mock_synthesizer,
        )

        # Should still complete without error
        result = await mode.execute(sample_request)
        assert isinstance(result, ChatCompletionResponse)

    async def test_ensemble_mode_preserves_model_field(
        self,
        mock_providers_agreeing: list[MagicMock],
        mock_synthesizer: MagicMock,
        sample_request: ChatCompletionRequest,
    ) -> None:
        """Response should preserve model field from synthesizer."""
        from src.orchestration.modes.ensemble import EnsembleMode

        mode = EnsembleMode(
            participants=mock_providers_agreeing,
            synthesizer=mock_synthesizer,
        )

        result = await mode.execute(sample_request)

        # Model should be from synthesizer
        assert result.model == "synthesizer"

    async def test_ensemble_mode_large_number_of_participants(
        self,
        mock_synthesizer: MagicMock,
        sample_request: ChatCompletionRequest,
    ) -> None:
        """Should work with many participants."""
        from src.orchestration.modes.ensemble import EnsembleMode

        providers = [
            create_mock_provider(
                f"model-{i}",
                create_mock_response(f"model-{i}", f"Response from model {i}"),
            )
            for i in range(5)
        ]

        mode = EnsembleMode(participants=providers, synthesizer=mock_synthesizer)
        result = await mode.execute(sample_request)

        assert isinstance(result, ChatCompletionResponse)
        assert result.orchestration is not None
        assert len(result.orchestration.models_used) == 6  # 5 participants + 1 synthesizer
