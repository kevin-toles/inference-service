"""Unit tests for Orchestrator dispatch.

Tests for:
- OrchestrationMode enum (AC-11.4)
- Orchestrator dispatches to correct mode (AC-11.4)
"""

import time
from collections.abc import AsyncIterator
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.models.requests import ChatCompletionRequest, Message
from src.models.responses import (
    ChatCompletionChunk,
    ChatCompletionResponse,
    Choice,
    ChoiceMessage,
    ChunkChoice,
    ChunkDelta,
    Usage,
)
from src.orchestration.orchestrator import (
    OrchestrationMode,
    Orchestrator,
    UnsupportedModeError,
)
from src.providers.base import InferenceProvider, ModelMetadata


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_provider() -> MagicMock:
    """Create a mock InferenceProvider."""
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
            Message(role="user", content="Hello!"),
        ],
    )


@pytest.fixture
def sample_response() -> ChatCompletionResponse:
    """Create a sample chat completion response."""
    return ChatCompletionResponse(
        id="chatcmpl-123",
        created=int(time.time()),
        model="phi-4",
        choices=[
            Choice(
                index=0,
                message=ChoiceMessage(
                    role="assistant",
                    content="Hello! How can I help?",
                ),
                finish_reason="stop",
            )
        ],
        usage=Usage(
            prompt_tokens=15,
            completion_tokens=8,
            total_tokens=23,
        ),
    )


@pytest.fixture
def sample_chunks() -> list[ChatCompletionChunk]:
    """Create sample streaming chunks."""
    return [
        ChatCompletionChunk(
            id="chatcmpl-123",
            created=int(time.time()),
            model="phi-4",
            choices=[
                ChunkChoice(
                    index=0,
                    delta=ChunkDelta(role="assistant"),
                    finish_reason=None,
                )
            ],
        ),
        ChatCompletionChunk(
            id="chatcmpl-123",
            created=int(time.time()),
            model="phi-4",
            choices=[
                ChunkChoice(
                    index=0,
                    delta=ChunkDelta(content="Hello"),
                    finish_reason="stop",
                )
            ],
        ),
    ]


# =============================================================================
# OrchestrationMode enum tests
# =============================================================================


class TestOrchestrationModeEnum:
    """Tests for OrchestrationMode enum."""

    def test_orchestration_mode_single_value(self) -> None:
        """Test SINGLE mode has correct value."""
        assert OrchestrationMode.SINGLE.value == "single"

    def test_orchestration_mode_critique_value(self) -> None:
        """Test CRITIQUE mode has correct value."""
        assert OrchestrationMode.CRITIQUE.value == "critique"

    def test_orchestration_mode_debate_value(self) -> None:
        """Test DEBATE mode has correct value."""
        assert OrchestrationMode.DEBATE.value == "debate"

    def test_orchestration_mode_ensemble_value(self) -> None:
        """Test ENSEMBLE mode has correct value."""
        assert OrchestrationMode.ENSEMBLE.value == "ensemble"

    def test_orchestration_mode_pipeline_value(self) -> None:
        """Test PIPELINE mode has correct value."""
        assert OrchestrationMode.PIPELINE.value == "pipeline"

    def test_orchestration_mode_from_string(self) -> None:
        """Test OrchestrationMode can be created from string."""
        assert OrchestrationMode("single") == OrchestrationMode.SINGLE
        assert OrchestrationMode("critique") == OrchestrationMode.CRITIQUE

    def test_orchestration_mode_all_values(self) -> None:
        """Test all expected modes are present."""
        expected_modes = {"single", "critique", "debate", "ensemble", "pipeline"}
        actual_modes = {mode.value for mode in OrchestrationMode}
        assert actual_modes == expected_modes


# =============================================================================
# AC-11.4: Orchestrator dispatches to correct mode
# =============================================================================


class TestOrchestratorInit:
    """Tests for Orchestrator initialization."""

    def test_orchestrator_init_with_provider(
        self, mock_provider: MagicMock
    ) -> None:
        """Test Orchestrator can be initialized with a provider."""
        orchestrator = Orchestrator(provider=mock_provider)

        assert orchestrator.provider is mock_provider

    def test_orchestrator_default_mode_is_single(
        self, mock_provider: MagicMock
    ) -> None:
        """Test Orchestrator defaults to single mode."""
        orchestrator = Orchestrator(provider=mock_provider)

        assert orchestrator.mode == OrchestrationMode.SINGLE

    def test_orchestrator_init_with_mode_string(
        self, mock_provider: MagicMock
    ) -> None:
        """Test Orchestrator can be initialized with mode as string."""
        orchestrator = Orchestrator(provider=mock_provider, mode="single")

        assert orchestrator.mode == OrchestrationMode.SINGLE

    def test_orchestrator_init_with_mode_enum(
        self, mock_provider: MagicMock
    ) -> None:
        """Test Orchestrator can be initialized with mode as enum."""
        orchestrator = Orchestrator(
            provider=mock_provider, mode=OrchestrationMode.SINGLE
        )

        assert orchestrator.mode == OrchestrationMode.SINGLE


class TestOrchestratorDispatch:
    """Tests for Orchestrator dispatch to correct mode."""

    @pytest.mark.asyncio
    async def test_orchestrator_dispatches_to_single_mode(
        self,
        mock_provider: MagicMock,
        sample_request: ChatCompletionRequest,
        sample_response: ChatCompletionResponse,
    ) -> None:
        """Test Orchestrator(mode='single') dispatches to SingleMode."""
        mock_provider.generate = AsyncMock(return_value=sample_response)

        orchestrator = Orchestrator(provider=mock_provider, mode="single")
        result = await orchestrator.execute(sample_request)

        # Verify SingleMode was used
        assert result.orchestration is not None
        assert result.orchestration.mode == "single"
        mock_provider.generate.assert_called_once()

    @pytest.mark.asyncio
    async def test_orchestrator_single_mode_returns_response(
        self,
        mock_provider: MagicMock,
        sample_request: ChatCompletionRequest,
        sample_response: ChatCompletionResponse,
    ) -> None:
        """Test Orchestrator returns valid response in single mode."""
        mock_provider.generate = AsyncMock(return_value=sample_response)

        orchestrator = Orchestrator(provider=mock_provider, mode="single")
        result = await orchestrator.execute(sample_request)

        assert result.choices[0].message.content == "Hello! How can I help?"
        assert result.usage.total_tokens == 23

    @pytest.mark.asyncio
    async def test_orchestrator_unsupported_mode_raises_error(
        self,
        mock_provider: MagicMock,
        sample_request: ChatCompletionRequest,
    ) -> None:
        """Test Orchestrator raises error for unsupported mode."""
        # critique/debate/etc. modes are not yet implemented
        orchestrator = Orchestrator(provider=mock_provider, mode="critique")

        with pytest.raises(UnsupportedModeError) as exc_info:
            await orchestrator.execute(sample_request)

        assert "critique" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_orchestrator_models_used_list(
        self,
        mock_provider: MagicMock,
        sample_request: ChatCompletionRequest,
        sample_response: ChatCompletionResponse,
    ) -> None:
        """Test Orchestrator response includes models_used list."""
        mock_provider.generate = AsyncMock(return_value=sample_response)

        orchestrator = Orchestrator(provider=mock_provider, mode="single")
        result = await orchestrator.execute(sample_request)

        assert result.orchestration is not None
        assert "phi-4" in result.orchestration.models_used


class TestOrchestratorStreaming:
    """Tests for Orchestrator streaming support."""

    @pytest.mark.asyncio
    async def test_orchestrator_stream_in_single_mode(
        self,
        mock_provider: MagicMock,
        sample_request: ChatCompletionRequest,
        sample_chunks: list[ChatCompletionChunk],
    ) -> None:
        """Test Orchestrator streaming in single mode."""

        async def mock_stream(
            request: ChatCompletionRequest,
        ) -> AsyncIterator[ChatCompletionChunk]:
            for chunk in sample_chunks:
                yield chunk

        mock_provider.stream = mock_stream

        orchestrator = Orchestrator(provider=mock_provider, mode="single")
        chunks_received = []

        async for chunk in orchestrator.stream(sample_request):
            chunks_received.append(chunk)

        assert len(chunks_received) == 2

    @pytest.mark.asyncio
    async def test_orchestrator_stream_unsupported_mode_raises(
        self,
        mock_provider: MagicMock,
        sample_request: ChatCompletionRequest,
    ) -> None:
        """Test Orchestrator stream raises error for unsupported mode."""
        orchestrator = Orchestrator(provider=mock_provider, mode="debate")

        with pytest.raises(UnsupportedModeError):
            async for _ in orchestrator.stream(sample_request):
                pass


# =============================================================================
# Edge cases
# =============================================================================


class TestOrchestratorEdgeCases:
    """Edge case tests for Orchestrator."""

    @pytest.mark.asyncio
    async def test_orchestrator_propagates_provider_errors(
        self,
        mock_provider: MagicMock,
        sample_request: ChatCompletionRequest,
    ) -> None:
        """Test Orchestrator propagates provider errors."""
        mock_provider.generate = AsyncMock(
            side_effect=RuntimeError("Provider error")
        )

        orchestrator = Orchestrator(provider=mock_provider, mode="single")

        with pytest.raises(RuntimeError, match="Provider error"):
            await orchestrator.execute(sample_request)

    def test_orchestrator_invalid_mode_string_raises(
        self, mock_provider: MagicMock
    ) -> None:
        """Test Orchestrator raises error for invalid mode string."""
        with pytest.raises(ValueError):
            Orchestrator(provider=mock_provider, mode="invalid_mode")

    @pytest.mark.asyncio
    async def test_orchestrator_with_different_provider(
        self,
        sample_request: ChatCompletionRequest,
    ) -> None:
        """Test Orchestrator works with different providers."""
        provider = MagicMock(spec=InferenceProvider)
        provider.model_info = ModelMetadata(
            model_id="llama-3.2-3b",
            context_length=8192,
            roles=["fast"],
            memory_mb=4000,
            status="loaded",
        )
        provider.is_loaded = True
        provider.generate = AsyncMock(
            return_value=ChatCompletionResponse(
                id="chatcmpl-456",
                created=int(time.time()),
                model="llama-3.2-3b",
                choices=[
                    Choice(
                        index=0,
                        message=ChoiceMessage(
                            role="assistant",
                            content="Quick response",
                        ),
                        finish_reason="stop",
                    )
                ],
                usage=Usage(
                    prompt_tokens=10,
                    completion_tokens=5,
                    total_tokens=15,
                ),
            )
        )

        orchestrator = Orchestrator(provider=provider, mode="single")
        result = await orchestrator.execute(sample_request)

        assert result.orchestration is not None
        assert result.orchestration.models_used == ["llama-3.2-3b"]


class TestUnsupportedModeError:
    """Tests for UnsupportedModeError exception."""

    def test_unsupported_mode_error_message(self) -> None:
        """Test UnsupportedModeError has informative message."""
        error = UnsupportedModeError("critique")

        assert "critique" in str(error)

    def test_unsupported_mode_error_is_exception(self) -> None:
        """Test UnsupportedModeError inherits from Exception."""
        error = UnsupportedModeError("debate")

        assert isinstance(error, Exception)
