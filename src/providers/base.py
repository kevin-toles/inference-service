"""Base classes for inference providers.

Defines the InferenceProvider ABC (Abstract Base Class) that all
concrete providers must implement.

Patterns applied:
- ABC with @abstractmethod decorator
- AsyncIterator for streaming (not AsyncGenerator)
- Dataclass with field(default_factory=list) for mutable defaults (AP-1.5)
- PEP 604 union syntax (X | None)

Reference: WBS-INF4 AC-4.1, AC-4.2, AC-4.3
"""

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass, field

from src.models.requests import ChatCompletionRequest
from src.models.responses import ChatCompletionChunk, ChatCompletionResponse


@dataclass
class ModelMetadata:
    """Model metadata information.

    Contains static information about a loaded model including
    its capabilities, resource requirements, and status.

    Attributes:
        model_id: Unique model identifier (e.g., "phi-4").
        context_length: Maximum context window in tokens.
        roles: List of roles this model supports (e.g., ["coder", "thinker"]).
        memory_mb: Memory requirement in megabytes.
        status: Current status ("available", "loaded", "loading").
        file_path: Path to model file (for local models).

    Note:
        Uses field(default_factory=list) for roles per AP-1.5
        to avoid mutable default argument bug.
    """

    model_id: str
    context_length: int
    roles: list[str] = field(default_factory=list)
    memory_mb: int = 0
    status: str = "available"
    file_path: str | None = None


class InferenceProvider(ABC):
    """Abstract base class for inference providers.

    Defines the interface that all inference providers must implement.
    This follows the Ports and Adapters (Hexagonal Architecture) pattern
    where InferenceProvider is the "port" and concrete implementations
    (LlamaCppProvider, etc.) are the "adapters".

    AC-4.1: Defines complete interface with generate, stream, tokenize, count_tokens.
    AC-4.2: Supports both sync (generate) and streaming (stream) generation.
    AC-4.3: Exposes model metadata via model_info property.

    Example:
        class MyProvider(InferenceProvider):
            async def generate(self, request):
                # Implementation
                ...

            async def stream(self, request):
                # Async generator implementation
                yield chunk
    """

    @property
    @abstractmethod
    def model_info(self) -> ModelMetadata:
        """Get model metadata.

        AC-4.3: Provider exposes model metadata (context length, roles).

        Returns:
            ModelMetadata with model information.
        """
        ...

    @property
    @abstractmethod
    def is_loaded(self) -> bool:
        """Check if model is currently loaded and ready.

        Returns:
            True if model is loaded and ready for inference.
        """
        ...

    @abstractmethod
    async def generate(
        self, request: ChatCompletionRequest
    ) -> ChatCompletionResponse:
        """Generate a non-streaming completion.

        AC-4.2: Provider supports sync generation.

        Args:
            request: Chat completion request with messages and parameters.

        Returns:
            Complete ChatCompletionResponse with choices and usage.

        Raises:
            ModelNotLoadedError: If model is not loaded.
            InferenceError: If generation fails.
        """
        ...

    @abstractmethod
    async def stream(
        self, request: ChatCompletionRequest
    ) -> AsyncIterator[ChatCompletionChunk]:
        """Generate a streaming completion.

        AC-4.2: Provider supports streaming generation.

        Yields ChatCompletionChunk objects as tokens are generated.
        The final chunk should have finish_reason set.

        Args:
            request: Chat completion request with messages and parameters.

        Yields:
            ChatCompletionChunk for each generated token/segment.

        Raises:
            ModelNotLoadedError: If model is not loaded.
            InferenceError: If generation fails.

        Note:
            This is an async generator - implementations should use
            `async for` to iterate and `yield` to produce chunks.
        """
        # AsyncIterator return type is correct for async generators
        # This is valid even without await - async generators work this way
        ...

    @abstractmethod
    def tokenize(self, text: str) -> list[int]:
        """Tokenize text into token IDs.

        Args:
            text: Text string to tokenize.

        Returns:
            List of token IDs.
        """
        ...

    @abstractmethod
    def count_tokens(self, text: str) -> int:
        """Count tokens in text.

        Args:
            text: Text string to count tokens in.

        Returns:
            Number of tokens.
        """
        ...

    async def load(self) -> None:  # noqa: B027
        """Load the model into memory.

        Override in subclass if model requires explicit loading.
        Default implementation does nothing (for providers that
        load on init).
        """

    async def unload(self) -> None:  # noqa: B027
        """Unload the model from memory.

        Override in subclass if model supports unloading.
        Default implementation does nothing.
        """
