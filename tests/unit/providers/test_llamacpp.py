"""Unit tests for LlamaCppProvider.

Tests the llama-cpp-python based inference provider.
Uses mocking to avoid requiring actual model files.

Reference: WBS-INF5 AC-5.1 through AC-5.6
"""

import asyncio
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.models.requests import ChatCompletionRequest, Message
from src.models.responses import ChatCompletionChunk, ChatCompletionResponse


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def mock_llama_class() -> MagicMock:
    """Create a mock Llama class."""
    mock = MagicMock()
    mock_instance = MagicMock()

    # Mock tokenize method
    mock_instance.tokenize.return_value = [1, 2, 3, 4, 5]  # 5 tokens

    # Mock create_chat_completion for non-streaming
    mock_instance.create_chat_completion.return_value = {
        "id": "chatcmpl-test123",
        "object": "chat.completion",
        "created": 1703704800,
        "model": "test-model",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "Test response"},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15,
        },
    }

    mock.return_value = mock_instance
    return mock


@pytest.fixture
def mock_llama_streaming() -> MagicMock:
    """Create a mock Llama class with streaming support."""
    mock = MagicMock()
    mock_instance = MagicMock()

    # Mock tokenize method
    mock_instance.tokenize.return_value = [1, 2, 3, 4, 5]

    def streaming_response(*args: Any, **kwargs: Any) -> Any:
        """Generate mock streaming chunks."""
        if kwargs.get("stream"):
            # Return iterator of chunks
            chunks = [
                {
                    "id": "chatcmpl-stream123",
                    "choices": [
                        {"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}
                    ],
                },
                {
                    "id": "chatcmpl-stream123",
                    "choices": [
                        {"index": 0, "delta": {"content": "Hello"}, "finish_reason": None}
                    ],
                },
                {
                    "id": "chatcmpl-stream123",
                    "choices": [
                        {"index": 0, "delta": {"content": " world"}, "finish_reason": None}
                    ],
                },
                {
                    "id": "chatcmpl-stream123",
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                },
            ]
            return iter(chunks)
        else:
            return {
                "id": "chatcmpl-test123",
                "object": "chat.completion",
                "created": 1703704800,
                "model": "test-model",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "Test response"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 5,
                    "total_tokens": 15,
                },
            }

    mock_instance.create_chat_completion.side_effect = streaming_response
    mock.return_value = mock_instance
    return mock


@pytest.fixture
def sample_request() -> ChatCompletionRequest:
    """Create a sample chat completion request."""
    return ChatCompletionRequest(
        model="phi-4",
        messages=[
            Message(role="system", content="You are a helpful assistant."),
            Message(role="user", content="Hello!"),
        ],
        max_tokens=100,
        temperature=0.7,
    )


@pytest.fixture
def model_path(tmp_path: Path) -> Path:
    """Create a mock model path."""
    model_file = tmp_path / "models" / "phi-4" / "phi-4-q4_k_m.gguf"
    model_file.parent.mkdir(parents=True)
    model_file.touch()
    return model_file


# =============================================================================
# TestLlamaCppProviderImport
# =============================================================================


class TestLlamaCppProviderImport:
    """Test that LlamaCppProvider can be imported."""

    def test_provider_importable(self) -> None:
        """AC-5.1: Provider class exists and can be imported."""
        from src.providers.llamacpp import LlamaCppProvider

        assert LlamaCppProvider is not None

    def test_provider_is_inference_provider(self) -> None:
        """AC-5.1: Provider inherits from InferenceProvider ABC."""
        from src.providers.base import InferenceProvider
        from src.providers.llamacpp import LlamaCppProvider

        assert issubclass(LlamaCppProvider, InferenceProvider)

    def test_provider_exceptions_importable(self) -> None:
        """AC-5.1: Custom exceptions can be imported."""
        from src.providers.llamacpp import (
            LlamaCppInferenceError,
            LlamaCppModelLoadError,
            LlamaCppModelNotFoundError,
            LlamaCppProviderError,
        )

        assert issubclass(LlamaCppModelNotFoundError, LlamaCppProviderError)
        assert issubclass(LlamaCppModelLoadError, LlamaCppProviderError)
        assert issubclass(LlamaCppInferenceError, LlamaCppProviderError)


# =============================================================================
# TestLlamaCppProviderInit
# =============================================================================


class TestLlamaCppProviderInit:
    """Test provider initialization."""

    def test_init_with_model_path(self, model_path: Path) -> None:
        """AC-5.1: Provider initializes with model path."""
        from src.providers.llamacpp import LlamaCppProvider

        provider = LlamaCppProvider(
            model_path=model_path,
            model_id="phi-4",
            context_length=4096,
        )

        assert provider is not None
        assert provider.model_info.model_id == "phi-4"
        assert provider.model_info.context_length == 4096

    def test_init_with_gpu_layers(self, model_path: Path) -> None:
        """AC-5.5: Provider accepts n_gpu_layers parameter."""
        from src.providers.llamacpp import LlamaCppProvider

        provider = LlamaCppProvider(
            model_path=model_path,
            model_id="phi-4",
            context_length=4096,
            n_gpu_layers=-1,  # All layers on GPU (Metal)
        )

        assert provider._n_gpu_layers == -1

    def test_init_with_roles(self, model_path: Path) -> None:
        """AC-5.1: Provider accepts roles list."""
        from src.providers.llamacpp import LlamaCppProvider

        provider = LlamaCppProvider(
            model_path=model_path,
            model_id="phi-4",
            context_length=4096,
            roles=["coder", "thinker"],
        )

        assert "coder" in provider.model_info.roles
        assert "thinker" in provider.model_info.roles

    def test_init_not_loaded_by_default(self, model_path: Path) -> None:
        """AC-5.6: Provider is not loaded until load() called."""
        from src.providers.llamacpp import LlamaCppProvider

        provider = LlamaCppProvider(
            model_path=model_path,
            model_id="phi-4",
            context_length=4096,
        )

        assert provider.is_loaded is False

    def test_init_with_invalid_path_raises_error(self) -> None:
        """AC-5.1: Invalid model path raises LlamaCppModelNotFoundError."""
        from src.providers.llamacpp import (
            LlamaCppModelNotFoundError,
            LlamaCppProvider,
        )

        with pytest.raises(LlamaCppModelNotFoundError):
            LlamaCppProvider(
                model_path=Path("/nonexistent/model.gguf"),
                model_id="test",
                context_length=4096,
            )


# =============================================================================
# TestLlamaCppProviderLoad
# =============================================================================


class TestLlamaCppProviderLoad:
    """Test model loading functionality."""

    @pytest.mark.asyncio
    async def test_load_model_success(
        self, model_path: Path, mock_llama_class: MagicMock
    ) -> None:
        """AC-5.1: load() successfully loads model into memory."""
        with patch("src.providers.llamacpp.Llama", mock_llama_class):
            from src.providers.llamacpp import LlamaCppProvider

            provider = LlamaCppProvider(
                model_path=model_path,
                model_id="phi-4",
                context_length=4096,
            )

            await provider.load()

            assert provider.is_loaded is True
            mock_llama_class.assert_called_once()

    @pytest.mark.asyncio
    async def test_load_with_metal_acceleration(
        self, model_path: Path, mock_llama_class: MagicMock
    ) -> None:
        """AC-5.5: load() uses n_gpu_layers=-1 for Metal on Mac."""
        with patch("src.providers.llamacpp.Llama", mock_llama_class):
            from src.providers.llamacpp import LlamaCppProvider

            provider = LlamaCppProvider(
                model_path=model_path,
                model_id="phi-4",
                context_length=4096,
                n_gpu_layers=-1,
            )

            await provider.load()

            # Verify Llama was called with n_gpu_layers=-1
            call_kwargs = mock_llama_class.call_args.kwargs
            assert call_kwargs.get("n_gpu_layers") == -1

    @pytest.mark.asyncio
    async def test_load_with_context_length(
        self, model_path: Path, mock_llama_class: MagicMock
    ) -> None:
        """AC-5.1: load() passes context_length as n_ctx."""
        with patch("src.providers.llamacpp.Llama", mock_llama_class):
            from src.providers.llamacpp import LlamaCppProvider

            provider = LlamaCppProvider(
                model_path=model_path,
                model_id="phi-4",
                context_length=8192,
            )

            await provider.load()

            call_kwargs = mock_llama_class.call_args.kwargs
            assert call_kwargs.get("n_ctx") == 8192

    @pytest.mark.asyncio
    async def test_load_failure_raises_error(
        self, model_path: Path
    ) -> None:
        """AC-5.6: load() raises LlamaCppModelLoadError on failure."""
        mock_llama = MagicMock()
        mock_llama.side_effect = RuntimeError("Metal initialization failed")

        with patch("src.providers.llamacpp.Llama", mock_llama):
            from src.providers.llamacpp import LlamaCppModelLoadError, LlamaCppProvider

            provider = LlamaCppProvider(
                model_path=model_path,
                model_id="phi-4",
                context_length=4096,
            )

            with pytest.raises(LlamaCppModelLoadError):
                await provider.load()

    @pytest.mark.asyncio
    async def test_load_idempotent(
        self, model_path: Path, mock_llama_class: MagicMock
    ) -> None:
        """AC-5.6: Multiple load() calls only load once."""
        with patch("src.providers.llamacpp.Llama", mock_llama_class):
            from src.providers.llamacpp import LlamaCppProvider

            provider = LlamaCppProvider(
                model_path=model_path,
                model_id="phi-4",
                context_length=4096,
            )

            await provider.load()
            await provider.load()  # Second call

            # Should only call Llama() once
            assert mock_llama_class.call_count == 1


# =============================================================================
# TestLlamaCppProviderUnload
# =============================================================================


class TestLlamaCppProviderUnload:
    """Test model unloading functionality."""

    @pytest.mark.asyncio
    async def test_unload_model(
        self, model_path: Path, mock_llama_class: MagicMock
    ) -> None:
        """AC-5.6: unload() removes model from memory."""
        with patch("src.providers.llamacpp.Llama", mock_llama_class):
            from src.providers.llamacpp import LlamaCppProvider

            provider = LlamaCppProvider(
                model_path=model_path,
                model_id="phi-4",
                context_length=4096,
            )

            await provider.load()
            assert provider.is_loaded is True

            await provider.unload()
            assert provider.is_loaded is False

    @pytest.mark.asyncio
    async def test_unload_when_not_loaded(self, model_path: Path) -> None:
        """AC-5.6: unload() when not loaded is a no-op."""
        from src.providers.llamacpp import LlamaCppProvider

        provider = LlamaCppProvider(
            model_path=model_path,
            model_id="phi-4",
            context_length=4096,
        )

        # Should not raise
        await provider.unload()
        assert provider.is_loaded is False


# =============================================================================
# TestLlamaCppProviderGenerate
# =============================================================================


class TestLlamaCppProviderGenerate:
    """Test non-streaming completion generation."""

    @pytest.mark.asyncio
    async def test_generate_returns_response(
        self,
        model_path: Path,
        mock_llama_class: MagicMock,
        sample_request: ChatCompletionRequest,
    ) -> None:
        """AC-5.2: generate() returns ChatCompletionResponse."""
        with patch("src.providers.llamacpp.Llama", mock_llama_class):
            from src.providers.llamacpp import LlamaCppProvider

            provider = LlamaCppProvider(
                model_path=model_path,
                model_id="phi-4",
                context_length=4096,
            )
            await provider.load()

            response = await provider.generate(sample_request)

            assert isinstance(response, ChatCompletionResponse)
            assert response.model == "phi-4"

    @pytest.mark.asyncio
    async def test_generate_returns_choices(
        self,
        model_path: Path,
        mock_llama_class: MagicMock,
        sample_request: ChatCompletionRequest,
    ) -> None:
        """AC-5.2: generate() response has choices with content."""
        with patch("src.providers.llamacpp.Llama", mock_llama_class):
            from src.providers.llamacpp import LlamaCppProvider

            provider = LlamaCppProvider(
                model_path=model_path,
                model_id="phi-4",
                context_length=4096,
            )
            await provider.load()

            response = await provider.generate(sample_request)

            assert len(response.choices) == 1
            assert response.choices[0].message.content == "Test response"
            assert response.choices[0].finish_reason == "stop"

    @pytest.mark.asyncio
    async def test_generate_returns_usage(
        self,
        model_path: Path,
        mock_llama_class: MagicMock,
        sample_request: ChatCompletionRequest,
    ) -> None:
        """AC-5.4: generate() response has correct usage statistics."""
        with patch("src.providers.llamacpp.Llama", mock_llama_class):
            from src.providers.llamacpp import LlamaCppProvider

            provider = LlamaCppProvider(
                model_path=model_path,
                model_id="phi-4",
                context_length=4096,
            )
            await provider.load()

            response = await provider.generate(sample_request)

            assert response.usage is not None
            assert response.usage.prompt_tokens == 10
            assert response.usage.completion_tokens == 5
            assert response.usage.total_tokens == 15

    @pytest.mark.asyncio
    async def test_generate_when_not_loaded_raises(
        self, model_path: Path, sample_request: ChatCompletionRequest
    ) -> None:
        """AC-5.2: generate() raises error when model not loaded."""
        from src.providers.llamacpp import LlamaCppInferenceError, LlamaCppProvider

        provider = LlamaCppProvider(
            model_path=model_path,
            model_id="phi-4",
            context_length=4096,
        )

        with pytest.raises(LlamaCppInferenceError):
            await provider.generate(sample_request)

    @pytest.mark.asyncio
    async def test_generate_passes_parameters(
        self,
        model_path: Path,
        mock_llama_class: MagicMock,
    ) -> None:
        """AC-5.2: generate() passes temperature, max_tokens to llama-cpp."""
        with patch("src.providers.llamacpp.Llama", mock_llama_class):
            from src.providers.llamacpp import LlamaCppProvider

            provider = LlamaCppProvider(
                model_path=model_path,
                model_id="phi-4",
                context_length=4096,
            )
            await provider.load()

            request = ChatCompletionRequest(
                model="phi-4",
                messages=[Message(role="user", content="Hello")],
                max_tokens=256,
                temperature=0.5,
                top_p=0.9,
            )

            await provider.generate(request)

            # Check parameters passed to create_chat_completion
            mock_instance = mock_llama_class.return_value
            call_kwargs = mock_instance.create_chat_completion.call_args.kwargs
            assert call_kwargs.get("max_tokens") == 256
            assert call_kwargs.get("temperature") == pytest.approx(0.5)
            assert call_kwargs.get("top_p") == pytest.approx(0.9)


# =============================================================================
# TestLlamaCppProviderStream
# =============================================================================


class TestLlamaCppProviderStream:
    """Test streaming completion generation."""

    @pytest.mark.asyncio
    async def test_stream_yields_chunks(
        self,
        model_path: Path,
        mock_llama_streaming: MagicMock,
        sample_request: ChatCompletionRequest,
    ) -> None:
        """AC-5.3: stream() yields ChatCompletionChunk objects."""
        with patch("src.providers.llamacpp.Llama", mock_llama_streaming):
            from src.providers.llamacpp import LlamaCppProvider

            provider = LlamaCppProvider(
                model_path=model_path,
                model_id="phi-4",
                context_length=4096,
            )
            await provider.load()

            chunks = []
            async for chunk in provider.stream(sample_request):
                chunks.append(chunk)

            assert len(chunks) > 0
            assert all(isinstance(c, ChatCompletionChunk) for c in chunks)

    @pytest.mark.asyncio
    async def test_stream_chunks_have_content(
        self,
        model_path: Path,
        mock_llama_streaming: MagicMock,
        sample_request: ChatCompletionRequest,
    ) -> None:
        """AC-5.3: Streaming chunks contain content deltas."""
        with patch("src.providers.llamacpp.Llama", mock_llama_streaming):
            from src.providers.llamacpp import LlamaCppProvider

            provider = LlamaCppProvider(
                model_path=model_path,
                model_id="phi-4",
                context_length=4096,
            )
            await provider.load()

            content_parts = []
            async for chunk in provider.stream(sample_request):
                if chunk.choices and chunk.choices[0].delta.content:
                    content_parts.append(chunk.choices[0].delta.content)

            assert "Hello" in content_parts
            assert " world" in content_parts

    @pytest.mark.asyncio
    async def test_stream_ends_with_finish_reason(
        self,
        model_path: Path,
        mock_llama_streaming: MagicMock,
        sample_request: ChatCompletionRequest,
    ) -> None:
        """AC-5.3: Final streaming chunk has finish_reason."""
        with patch("src.providers.llamacpp.Llama", mock_llama_streaming):
            from src.providers.llamacpp import LlamaCppProvider

            provider = LlamaCppProvider(
                model_path=model_path,
                model_id="phi-4",
                context_length=4096,
            )
            await provider.load()

            last_chunk = None
            async for chunk in provider.stream(sample_request):
                last_chunk = chunk

            assert last_chunk is not None
            assert last_chunk.choices[0].finish_reason == "stop"

    @pytest.mark.asyncio
    async def test_stream_when_not_loaded_raises(
        self, model_path: Path, sample_request: ChatCompletionRequest
    ) -> None:
        """AC-5.3: stream() raises error when model not loaded."""
        from src.providers.llamacpp import LlamaCppInferenceError, LlamaCppProvider

        provider = LlamaCppProvider(
            model_path=model_path,
            model_id="phi-4",
            context_length=4096,
        )

        with pytest.raises(LlamaCppInferenceError):
            async for _ in provider.stream(sample_request):
                pass  # Error expected when model not loaded

    @pytest.mark.asyncio
    async def test_stream_uses_stream_parameter(
        self,
        model_path: Path,
        mock_llama_streaming: MagicMock,
        sample_request: ChatCompletionRequest,
    ) -> None:
        """AC-5.3: stream() passes stream=True to llama-cpp."""
        with patch("src.providers.llamacpp.Llama", mock_llama_streaming):
            from src.providers.llamacpp import LlamaCppProvider

            provider = LlamaCppProvider(
                model_path=model_path,
                model_id="phi-4",
                context_length=4096,
            )
            await provider.load()

            async for _ in provider.stream(sample_request):
                pass  # Consume stream to verify parameters

            mock_instance = mock_llama_streaming.return_value
            call_kwargs = mock_instance.create_chat_completion.call_args.kwargs
            assert call_kwargs.get("stream") is True


# =============================================================================
# TestLlamaCppProviderTokenize
# =============================================================================


class TestLlamaCppProviderTokenize:
    """Test tokenization functionality."""

    @pytest.mark.asyncio
    async def test_tokenize_returns_token_ids(
        self, model_path: Path, mock_llama_class: MagicMock
    ) -> None:
        """AC-5.4: tokenize() returns list of token IDs."""
        with patch("src.providers.llamacpp.Llama", mock_llama_class):
            from src.providers.llamacpp import LlamaCppProvider

            provider = LlamaCppProvider(
                model_path=model_path,
                model_id="phi-4",
                context_length=4096,
            )
            await provider.load()

            tokens = provider.tokenize("Hello world")

            assert isinstance(tokens, list)
            assert all(isinstance(t, int) for t in tokens)

    @pytest.mark.asyncio
    async def test_tokenize_calls_model_tokenize(
        self, model_path: Path, mock_llama_class: MagicMock
    ) -> None:
        """AC-5.4: tokenize() uses model's tokenizer."""
        with patch("src.providers.llamacpp.Llama", mock_llama_class):
            from src.providers.llamacpp import LlamaCppProvider

            provider = LlamaCppProvider(
                model_path=model_path,
                model_id="phi-4",
                context_length=4096,
            )
            await provider.load()

            provider.tokenize("Hello world")

            mock_instance = mock_llama_class.return_value
            mock_instance.tokenize.assert_called_once()
            # Should be called with bytes
            call_args = mock_instance.tokenize.call_args[0][0]
            assert isinstance(call_args, bytes)


class TestLlamaCppProviderCountTokens:
    """Test token counting functionality."""

    @pytest.mark.asyncio
    async def test_count_tokens_returns_int(
        self, model_path: Path, mock_llama_class: MagicMock
    ) -> None:
        """AC-5.4: count_tokens() returns integer count."""
        with patch("src.providers.llamacpp.Llama", mock_llama_class):
            from src.providers.llamacpp import LlamaCppProvider

            provider = LlamaCppProvider(
                model_path=model_path,
                model_id="phi-4",
                context_length=4096,
            )
            await provider.load()

            count = provider.count_tokens("Hello world")

            assert isinstance(count, int)
            assert count == 5  # Mock returns [1, 2, 3, 4, 5]

    def test_count_tokens_when_not_loaded_uses_estimate(
        self, model_path: Path
    ) -> None:
        """AC-5.4: count_tokens() uses estimate when model not loaded."""
        from src.providers.llamacpp import LlamaCppProvider

        provider = LlamaCppProvider(
            model_path=model_path,
            model_id="phi-4",
            context_length=4096,
        )

        # Should not raise, uses heuristic
        count = provider.count_tokens("Hello world")
        assert isinstance(count, int)
        assert count > 0


# =============================================================================
# TestLlamaCppProviderModelInfo
# =============================================================================


class TestLlamaCppProviderModelInfo:
    """Test model metadata access."""

    def test_model_info_returns_metadata(self, model_path: Path) -> None:
        """AC-5.1: model_info returns ModelMetadata."""
        from src.providers.base import ModelMetadata
        from src.providers.llamacpp import LlamaCppProvider

        provider = LlamaCppProvider(
            model_path=model_path,
            model_id="phi-4",
            context_length=16384,
            roles=["coder"],
        )

        info = provider.model_info

        assert isinstance(info, ModelMetadata)
        assert info.model_id == "phi-4"
        assert info.context_length == 16384
        assert "coder" in info.roles

    def test_model_info_includes_file_path(self, model_path: Path) -> None:
        """AC-5.1: model_info includes file_path."""
        from src.providers.llamacpp import LlamaCppProvider

        provider = LlamaCppProvider(
            model_path=model_path,
            model_id="phi-4",
            context_length=4096,
        )

        info = provider.model_info

        assert info.file_path == str(model_path)

    @pytest.mark.asyncio
    async def test_model_info_status_updates(
        self, model_path: Path, mock_llama_class: MagicMock
    ) -> None:
        """AC-5.1: model_info.status reflects loaded state."""
        with patch("src.providers.llamacpp.Llama", mock_llama_class):
            from src.providers.llamacpp import LlamaCppProvider

            provider = LlamaCppProvider(
                model_path=model_path,
                model_id="phi-4",
                context_length=4096,
            )

            assert provider.model_info.status == "available"

            await provider.load()
            assert provider.model_info.status == "loaded"

            await provider.unload()
            assert provider.model_info.status == "available"


# =============================================================================
# TestLlamaCppProviderExceptions
# =============================================================================


class TestLlamaCppProviderExceptions:
    """Test exception handling."""

    def test_exception_names_end_in_error(self) -> None:
        """AP-7: All exception class names end in 'Error'."""
        from src.providers.llamacpp import (
            LlamaCppInferenceError,
            LlamaCppModelLoadError,
            LlamaCppModelNotFoundError,
            LlamaCppProviderError,
        )

        for exc_class in [
            LlamaCppProviderError,
            LlamaCppModelNotFoundError,
            LlamaCppModelLoadError,
            LlamaCppInferenceError,
        ]:
            assert exc_class.__name__.endswith("Error")

    def test_exceptions_have_hierarchy(self) -> None:
        """AC-5.2: Exceptions form proper hierarchy."""
        from src.providers.llamacpp import (
            LlamaCppInferenceError,
            LlamaCppModelLoadError,
            LlamaCppModelNotFoundError,
            LlamaCppProviderError,
        )

        assert issubclass(LlamaCppModelNotFoundError, LlamaCppProviderError)
        assert issubclass(LlamaCppModelLoadError, LlamaCppProviderError)
        assert issubclass(LlamaCppInferenceError, LlamaCppProviderError)
        assert issubclass(LlamaCppProviderError, Exception)


# =============================================================================
# TestLlamaCppProviderContextManager
# =============================================================================


class TestLlamaCppProviderContextManager:
    """Test context manager support."""

    @pytest.mark.asyncio
    async def test_async_context_manager(
        self, model_path: Path, mock_llama_class: MagicMock
    ) -> None:
        """AC-5.6: Provider supports async context manager."""
        with patch("src.providers.llamacpp.Llama", mock_llama_class):
            from src.providers.llamacpp import LlamaCppProvider

            provider = LlamaCppProvider(
                model_path=model_path,
                model_id="phi-4",
                context_length=4096,
            )

            async with provider:
                assert provider.is_loaded is True

            assert provider.is_loaded is False

    @pytest.mark.asyncio
    async def test_context_manager_handles_exception(
        self, model_path: Path, mock_llama_class: MagicMock
    ) -> None:
        """AC-5.6: Context manager unloads on exception."""
        with patch("src.providers.llamacpp.Llama", mock_llama_class):
            from src.providers.llamacpp import LlamaCppProvider

            provider = LlamaCppProvider(
                model_path=model_path,
                model_id="phi-4",
                context_length=4096,
            )

            with pytest.raises(ValueError):
                async with provider:
                    assert provider.is_loaded is True
                    raise ValueError("Test error")

            # Should still unload
            assert provider.is_loaded is False
