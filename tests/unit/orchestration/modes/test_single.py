"""Unit tests for SingleMode orchestration.

Tests for:
- SingleMode passing request to one model (AC-11.1)
- SingleMode returning response with usage stats (AC-11.2)
- SingleMode streaming support (AC-11.3)
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
from src.orchestration.modes.single import SingleMode
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
            Message(role="user", content="Hello, world!"),
        ],
        temperature=0.7,
        max_tokens=100,
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
                    content="Hello! How can I help you today?",
                ),
                finish_reason="stop",
            )
        ],
        usage=Usage(
            prompt_tokens=20,
            completion_tokens=10,
            total_tokens=30,
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
                    delta=ChunkDelta(content="!"),
                    finish_reason="stop",
                )
            ],
        ),
    ]


# =============================================================================
# AC-11.1: SingleMode passes request directly to one model
# =============================================================================


class TestSingleModeExecution:
    """Tests for SingleMode execute() method."""

    @pytest.mark.asyncio
    async def test_single_mode_passes_request_to_provider(
        self,
        mock_provider: MagicMock,
        sample_request: ChatCompletionRequest,
        sample_response: ChatCompletionResponse,
    ) -> None:
        """Test SingleMode passes request directly to provider.generate()."""
        mock_provider.generate = AsyncMock(return_value=sample_response)

        mode = SingleMode(provider=mock_provider)
        await mode.execute(sample_request)

        mock_provider.generate.assert_called_once_with(sample_request)

    @pytest.mark.asyncio
    async def test_single_mode_uses_one_model_only(
        self,
        mock_provider: MagicMock,
        sample_request: ChatCompletionRequest,
        sample_response: ChatCompletionResponse,
    ) -> None:
        """Test SingleMode uses exactly one model."""
        mock_provider.generate = AsyncMock(return_value=sample_response)

        mode = SingleMode(provider=mock_provider)
        result = await mode.execute(sample_request)

        # Verify only one model was used
        assert result.orchestration is not None
        assert len(result.orchestration.models_used) == 1
        assert result.orchestration.models_used[0] == "phi-4"

    @pytest.mark.asyncio
    async def test_single_mode_returns_response_unchanged(
        self,
        mock_provider: MagicMock,
        sample_request: ChatCompletionRequest,
        sample_response: ChatCompletionResponse,
    ) -> None:
        """Test SingleMode returns provider response content unchanged."""
        mock_provider.generate = AsyncMock(return_value=sample_response)

        mode = SingleMode(provider=mock_provider)
        result = await mode.execute(sample_request)

        # Verify content is unchanged
        assert result.choices[0].message.content == "Hello! How can I help you today?"
        assert result.model == "phi-4"

    @pytest.mark.asyncio
    async def test_single_mode_mode_is_single(
        self,
        mock_provider: MagicMock,
        sample_request: ChatCompletionRequest,
        sample_response: ChatCompletionResponse,
    ) -> None:
        """Test SingleMode sets orchestration mode to 'single'."""
        mock_provider.generate = AsyncMock(return_value=sample_response)

        mode = SingleMode(provider=mock_provider)
        result = await mode.execute(sample_request)

        assert result.orchestration is not None
        assert result.orchestration.mode == "single"


# =============================================================================
# AC-11.2: SingleMode returns response with usage stats
# =============================================================================


class TestSingleModeUsageStats:
    """Tests for SingleMode usage statistics."""

    @pytest.mark.asyncio
    async def test_single_mode_preserves_usage_stats(
        self,
        mock_provider: MagicMock,
        sample_request: ChatCompletionRequest,
        sample_response: ChatCompletionResponse,
    ) -> None:
        """Test SingleMode preserves usage statistics from provider."""
        mock_provider.generate = AsyncMock(return_value=sample_response)

        mode = SingleMode(provider=mock_provider)
        result = await mode.execute(sample_request)

        assert result.usage.prompt_tokens == 20
        assert result.usage.completion_tokens == 10
        assert result.usage.total_tokens == 30

    @pytest.mark.asyncio
    async def test_single_mode_includes_inference_time(
        self,
        mock_provider: MagicMock,
        sample_request: ChatCompletionRequest,
        sample_response: ChatCompletionResponse,
    ) -> None:
        """Test SingleMode includes inference time in metadata."""

        async def slow_generate(request: ChatCompletionRequest) -> ChatCompletionResponse:
            # Simulate some inference time
            await asyncio.sleep(0.01)  # 10ms
            return sample_response

        import asyncio

        mock_provider.generate = slow_generate

        mode = SingleMode(provider=mock_provider)
        result = await mode.execute(sample_request)

        assert result.orchestration is not None
        assert result.orchestration.total_inference_time_ms > 0

    @pytest.mark.asyncio
    async def test_single_mode_rounds_is_none_for_single(
        self,
        mock_provider: MagicMock,
        sample_request: ChatCompletionRequest,
        sample_response: ChatCompletionResponse,
    ) -> None:
        """Test SingleMode sets rounds to None (not applicable)."""
        mock_provider.generate = AsyncMock(return_value=sample_response)

        mode = SingleMode(provider=mock_provider)
        result = await mode.execute(sample_request)

        assert result.orchestration is not None
        assert result.orchestration.rounds is None

    @pytest.mark.asyncio
    async def test_single_mode_scores_are_none(
        self,
        mock_provider: MagicMock,
        sample_request: ChatCompletionRequest,
        sample_response: ChatCompletionResponse,
    ) -> None:
        """Test SingleMode sets scores to None (not applicable for single)."""
        mock_provider.generate = AsyncMock(return_value=sample_response)

        mode = SingleMode(provider=mock_provider)
        result = await mode.execute(sample_request)

        assert result.orchestration is not None
        assert result.orchestration.final_score is None
        assert result.orchestration.agreement_score is None


# =============================================================================
# AC-11.3: SingleMode supports streaming
# =============================================================================


class TestSingleModeStreaming:
    """Tests for SingleMode streaming support."""

    @pytest.mark.asyncio
    async def test_single_mode_stream_yields_chunks(
        self,
        mock_provider: MagicMock,
        sample_request: ChatCompletionRequest,
        sample_chunks: list[ChatCompletionChunk],
    ) -> None:
        """Test SingleMode stream() yields chunks from provider."""

        async def mock_stream(
            request: ChatCompletionRequest,
        ) -> AsyncIterator[ChatCompletionChunk]:
            for chunk in sample_chunks:
                yield chunk

        mock_provider.stream = mock_stream

        mode = SingleMode(provider=mock_provider)
        chunks_received = []

        async for chunk in mode.stream(sample_request):
            chunks_received.append(chunk)

        assert len(chunks_received) == 3

    @pytest.mark.asyncio
    async def test_single_mode_stream_preserves_chunk_content(
        self,
        mock_provider: MagicMock,
        sample_request: ChatCompletionRequest,
        sample_chunks: list[ChatCompletionChunk],
    ) -> None:
        """Test SingleMode stream preserves chunk content."""

        async def mock_stream(
            request: ChatCompletionRequest,
        ) -> AsyncIterator[ChatCompletionChunk]:
            for chunk in sample_chunks:
                yield chunk

        mock_provider.stream = mock_stream

        mode = SingleMode(provider=mock_provider)
        chunks_received = []

        async for chunk in mode.stream(sample_request):
            chunks_received.append(chunk)

        # First chunk has role
        assert chunks_received[0].choices[0].delta.role == "assistant"
        # Second chunk has content
        assert chunks_received[1].choices[0].delta.content == "Hello"
        # Third chunk has finish_reason
        assert chunks_received[2].choices[0].finish_reason == "stop"

    @pytest.mark.asyncio
    async def test_single_mode_stream_passes_request_to_provider(
        self,
        mock_provider: MagicMock,
        sample_request: ChatCompletionRequest,
        sample_chunks: list[ChatCompletionChunk],
    ) -> None:
        """Test SingleMode stream passes request to provider.stream()."""
        received_request = None

        async def mock_stream(
            request: ChatCompletionRequest,
        ) -> AsyncIterator[ChatCompletionChunk]:
            nonlocal received_request
            received_request = request
            for chunk in sample_chunks:
                yield chunk

        mock_provider.stream = mock_stream

        mode = SingleMode(provider=mock_provider)

        async for _ in mode.stream(sample_request):
            pass  # Consume stream to verify request forwarding

        assert received_request is sample_request

    @pytest.mark.asyncio
    async def test_single_mode_stream_handles_empty_stream(
        self,
        mock_provider: MagicMock,
        sample_request: ChatCompletionRequest,
    ) -> None:
        """Test SingleMode stream handles empty stream gracefully."""

        async def mock_stream(
            request: ChatCompletionRequest,  # noqa: ARG001
        ) -> AsyncIterator[ChatCompletionChunk]:
            # Empty generator pattern for testing empty streams
            for _ in []:
                yield ChatCompletionChunk(  # pragma: no cover
                    id="", choices=[], model="", created=0, object=""
                )

        mock_provider.stream = mock_stream

        mode = SingleMode(provider=mock_provider)
        chunks_received = []

        async for chunk in mode.stream(sample_request):
            chunks_received.append(chunk)

        assert len(chunks_received) == 0


# =============================================================================
# SingleMode initialization tests
# =============================================================================


class TestSingleModeInit:
    """Tests for SingleMode initialization."""

    def test_single_mode_init_with_provider(
        self, mock_provider: MagicMock
    ) -> None:
        """Test SingleMode can be initialized with a provider."""
        mode = SingleMode(provider=mock_provider)

        assert mode.provider is mock_provider

    def test_single_mode_model_id_from_provider(
        self, mock_provider: MagicMock
    ) -> None:
        """Test SingleMode gets model_id from provider."""
        mode = SingleMode(provider=mock_provider)

        assert mode.model_id == "phi-4"


# =============================================================================
# Edge cases
# =============================================================================


class TestSingleModeEdgeCases:
    """Edge case tests for SingleMode."""

    @pytest.mark.asyncio
    async def test_single_mode_handles_provider_error(
        self,
        mock_provider: MagicMock,
        sample_request: ChatCompletionRequest,
    ) -> None:
        """Test SingleMode propagates provider errors."""
        mock_provider.generate = AsyncMock(
            side_effect=RuntimeError("Provider error")
        )

        mode = SingleMode(provider=mock_provider)

        with pytest.raises(RuntimeError, match="Provider error"):
            await mode.execute(sample_request)

    @pytest.mark.asyncio
    async def test_single_mode_stream_handles_provider_error(
        self,
        mock_provider: MagicMock,
        sample_request: ChatCompletionRequest,
    ) -> None:
        """Test SingleMode stream propagates provider errors."""

        async def error_stream(
            request: ChatCompletionRequest,  # noqa: ARG001
        ) -> AsyncIterator[ChatCompletionChunk]:
            # First yield nothing, then raise
            for _ in []:
                yield ChatCompletionChunk(  # pragma: no cover
                    id="", choices=[], model="", created=0, object=""
                )
            raise RuntimeError("Stream error")

        mock_provider.stream = error_stream

        mode = SingleMode(provider=mock_provider)

        with pytest.raises(RuntimeError, match="Stream error"):
            async for _ in mode.stream(sample_request):
                pass  # Error expected before iteration completes

    @pytest.mark.asyncio
    async def test_single_mode_with_different_model(
        self,
        sample_request: ChatCompletionRequest,
        sample_response: ChatCompletionResponse,
    ) -> None:
        """Test SingleMode works with different model providers."""
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

        mode = SingleMode(provider=provider)
        result = await mode.execute(sample_request)

        assert result.orchestration is not None
        assert result.orchestration.models_used == ["llama-3.2-3b"]
        assert result.model == "llama-3.2-3b"
