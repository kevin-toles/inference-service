"""Tests for InferenceProvider ABC and provider interface.

TDD RED Phase: These tests define the expected behavior of the provider interface.
Reference: WBS-INF4 AC-4.1, AC-4.2, AC-4.3, AC-4.4

Tests verify:
- AC-4.1: InferenceProvider ABC defines complete interface
- AC-4.2: Provider supports both sync and streaming generation
- AC-4.3: Provider exposes model metadata (context length, roles)
- AC-4.4: Provider interface is testable with mock implementation
"""

from typing import AsyncIterator

import pytest

from src.models.requests import ChatCompletionRequest, Message
from src.models.responses import ChatCompletionChunk, ChatCompletionResponse


class TestInferenceProviderABC:
    """Test InferenceProvider ABC definition."""

    def test_inference_provider_is_abc(self) -> None:
        """InferenceProvider is an abstract base class."""
        from abc import ABC

        from src.providers.base import InferenceProvider

        assert issubclass(InferenceProvider, ABC)

    def test_cannot_instantiate_abc_directly(self) -> None:
        """ABC prevents direct instantiation.
        
        AC-4.1: ABC prevents instantiation without implementing all methods.
        """
        from src.providers.base import InferenceProvider

        with pytest.raises(TypeError) as exc_info:
            InferenceProvider()  # type: ignore[abstract]
        assert "abstract" in str(exc_info.value).lower()

    def test_has_generate_abstract_method(self) -> None:
        """InferenceProvider defines generate() abstract method."""
        from src.providers.base import InferenceProvider

        assert hasattr(InferenceProvider, "generate")
        # Check it's abstract
        assert getattr(InferenceProvider.generate, "__isabstractmethod__", False)

    def test_has_stream_abstract_method(self) -> None:
        """InferenceProvider defines stream() abstract method."""
        from src.providers.base import InferenceProvider

        assert hasattr(InferenceProvider, "stream")
        assert getattr(InferenceProvider.stream, "__isabstractmethod__", False)

    def test_has_model_info_property(self) -> None:
        """InferenceProvider has model_info property.
        
        AC-4.3: Provider exposes model metadata.
        """
        from src.providers.base import InferenceProvider

        assert hasattr(InferenceProvider, "model_info")

    def test_has_tokenize_abstract_method(self) -> None:
        """InferenceProvider defines tokenize() abstract method."""
        from src.providers.base import InferenceProvider

        assert hasattr(InferenceProvider, "tokenize")
        assert getattr(InferenceProvider.tokenize, "__isabstractmethod__", False)

    def test_has_count_tokens_abstract_method(self) -> None:
        """InferenceProvider defines count_tokens() abstract method."""
        from src.providers.base import InferenceProvider

        assert hasattr(InferenceProvider, "count_tokens")
        assert getattr(InferenceProvider.count_tokens, "__isabstractmethod__", False)

    def test_has_is_loaded_property(self) -> None:
        """InferenceProvider has is_loaded property."""
        from src.providers.base import InferenceProvider

        assert hasattr(InferenceProvider, "is_loaded")


class TestModelMetadata:
    """Test ModelMetadata dataclass."""

    def test_model_metadata_exists(self) -> None:
        """ModelMetadata dataclass is importable."""
        from src.providers.base import ModelMetadata

        assert ModelMetadata is not None

    def test_model_metadata_has_required_fields(self) -> None:
        """ModelMetadata has all required fields."""
        from src.providers.base import ModelMetadata

        meta = ModelMetadata(
            model_id="phi-4",
            context_length=16384,
        )
        assert meta.model_id == "phi-4"
        assert meta.context_length == 16384

    def test_model_metadata_has_roles(self) -> None:
        """ModelMetadata has roles field.
        
        AC-4.3: Provider exposes model metadata (context length, roles).
        """
        from src.providers.base import ModelMetadata

        meta = ModelMetadata(
            model_id="phi-4",
            context_length=16384,
            roles=["primary", "coder", "thinker"],
        )
        assert meta.roles == ["primary", "coder", "thinker"]

    def test_model_metadata_roles_default_empty(self) -> None:
        """ModelMetadata roles defaults to empty list.
        
        Uses field(default_factory=list) per AP-1.5.
        """
        from src.providers.base import ModelMetadata

        meta1 = ModelMetadata(model_id="a", context_length=1000)
        meta2 = ModelMetadata(model_id="b", context_length=2000)
        # Ensure they don't share the same list (mutable default bug)
        meta1.roles.append("test")
        assert "test" not in meta2.roles

    def test_model_metadata_optional_fields(self) -> None:
        """ModelMetadata has optional fields with defaults."""
        from src.providers.base import ModelMetadata

        meta = ModelMetadata(model_id="phi-4", context_length=16384)
        assert meta.memory_mb == 0
        assert meta.status == "available"
        assert meta.file_path is None


class TestProviderWithMock:
    """Test provider interface with mock implementation.
    
    AC-4.4: Provider interface is testable with mock implementation.
    """

    def test_mock_provider_importable(self) -> None:
        """MockProvider is importable from test fixtures."""
        from tests.unit.providers.mock_provider import MockProvider

        assert MockProvider is not None

    def test_mock_provider_is_inference_provider(self) -> None:
        """MockProvider implements InferenceProvider."""
        from src.providers.base import InferenceProvider
        from tests.unit.providers.mock_provider import MockProvider

        assert issubclass(MockProvider, InferenceProvider)

    def test_mock_provider_instantiates(self) -> None:
        """MockProvider can be instantiated."""
        from tests.unit.providers.mock_provider import MockProvider

        provider = MockProvider(model_id="test-model")
        assert provider is not None

    def test_mock_provider_has_model_info(self) -> None:
        """MockProvider exposes model_info."""
        from tests.unit.providers.mock_provider import MockProvider

        provider = MockProvider(model_id="test-model", context_length=4096)
        info = provider.model_info
        assert info.model_id == "test-model"
        assert info.context_length == 4096

    @pytest.mark.asyncio
    async def test_mock_provider_generate(self) -> None:
        """MockProvider.generate() returns ChatCompletionResponse.
        
        AC-4.2: Provider supports sync generation.
        """
        from tests.unit.providers.mock_provider import MockProvider

        provider = MockProvider(model_id="test-model")
        request = ChatCompletionRequest(
            model="test-model",
            messages=[Message(role="user", content="Hello")],
        )
        response = await provider.generate(request)
        assert isinstance(response, ChatCompletionResponse)
        assert len(response.choices) > 0

    @pytest.mark.asyncio
    async def test_mock_provider_stream(self) -> None:
        """MockProvider.stream() yields ChatCompletionChunk.
        
        AC-4.2: Provider supports streaming generation.
        """
        from tests.unit.providers.mock_provider import MockProvider

        provider = MockProvider(model_id="test-model")
        request = ChatCompletionRequest(
            model="test-model",
            messages=[Message(role="user", content="Hello")],
        )
        chunks: list[ChatCompletionChunk] = []
        async for chunk in provider.stream(request):
            chunks.append(chunk)

        assert len(chunks) > 0
        assert all(isinstance(c, ChatCompletionChunk) for c in chunks)

    def test_mock_provider_tokenize(self) -> None:
        """MockProvider.tokenize() returns token list."""
        from tests.unit.providers.mock_provider import MockProvider

        provider = MockProvider(model_id="test-model")
        tokens = provider.tokenize("Hello world")
        assert isinstance(tokens, list)
        assert len(tokens) > 0

    def test_mock_provider_count_tokens(self) -> None:
        """MockProvider.count_tokens() returns token count."""
        from tests.unit.providers.mock_provider import MockProvider

        provider = MockProvider(model_id="test-model")
        count = provider.count_tokens("Hello world")
        assert isinstance(count, int)
        assert count > 0

    def test_mock_provider_is_loaded(self) -> None:
        """MockProvider.is_loaded returns boolean."""
        from tests.unit.providers.mock_provider import MockProvider

        provider = MockProvider(model_id="test-model")
        assert isinstance(provider.is_loaded, bool)


class TestProviderTypeAnnotations:
    """Test provider type annotations for mypy compliance."""

    def test_generate_return_type_annotated(self) -> None:
        """generate() has proper return type annotation."""
        from src.providers.base import InferenceProvider

        # Get type hints
        import typing

        hints = typing.get_type_hints(InferenceProvider.generate)
        assert "return" in hints

    def test_stream_return_type_is_async_iterator(self) -> None:
        """stream() returns AsyncIterator[ChatCompletionChunk]."""
        from src.providers.base import InferenceProvider

        import typing
        from collections.abc import AsyncIterator as ABCAsyncIterator

        hints = typing.get_type_hints(InferenceProvider.stream)
        return_type = hints.get("return")
        # Check it's an AsyncIterator
        assert return_type is not None
        origin = typing.get_origin(return_type)
        # Handle both typing.AsyncIterator and collections.abc.AsyncIterator
        assert origin is AsyncIterator or origin is ABCAsyncIterator
