"""Mock provider for testing InferenceProvider interface.

Provides a concrete implementation of InferenceProvider for use in tests.
This validates that the ABC interface is correctly defined and usable.
"""

import time
from typing import AsyncIterator

from src.models.requests import ChatCompletionRequest
from src.models.responses import (
    ChatCompletionChunk,
    ChatCompletionResponse,
    Choice,
    ChoiceMessage,
    ChunkChoice,
    ChunkDelta,
    Usage,
)
from src.providers.base import InferenceProvider, ModelMetadata


class MockProvider(InferenceProvider):
    """Mock implementation of InferenceProvider for testing.

    Provides deterministic responses for validating interface contract.
    """

    def __init__(
        self,
        model_id: str = "mock-model",
        context_length: int = 4096,
        roles: list[str] | None = None,
    ) -> None:
        """Initialize mock provider.

        Args:
            model_id: Model identifier to return in metadata.
            context_length: Context length to return in metadata.
            roles: List of roles to return in metadata.
        """
        self._model_id = model_id
        self._context_length = context_length
        self._roles = roles if roles is not None else ["coder"]
        self._is_loaded = True

    @property
    def model_info(self) -> ModelMetadata:
        """Get mock model metadata."""
        return ModelMetadata(
            model_id=self._model_id,
            context_length=self._context_length,
            roles=self._roles.copy(),
            memory_mb=1024,
            status="loaded" if self._is_loaded else "available",
        )

    @property
    def is_loaded(self) -> bool:
        """Check if mock model is loaded."""
        return self._is_loaded

    async def generate(
        self, request: ChatCompletionRequest
    ) -> ChatCompletionResponse:
        """Generate mock completion response."""
        return ChatCompletionResponse(
            id="mock-response-id",
            model=self._model_id,
            created=int(time.time()),
            choices=[
                Choice(
                    index=0,
                    message=ChoiceMessage(
                        role="assistant",
                        content="Mock response content",
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

    async def stream(
        self, request: ChatCompletionRequest
    ) -> AsyncIterator[ChatCompletionChunk]:
        """Generate mock streaming response."""
        created = int(time.time())

        # Yield content chunks
        for word in ["Mock", " ", "streaming", " ", "response"]:
            yield ChatCompletionChunk(
                id="mock-chunk-id",
                model=self._model_id,
                created=created,
                choices=[
                    ChunkChoice(
                        index=0,
                        delta=ChunkDelta(content=word),
                        finish_reason=None,
                    )
                ],
            )

        # Final chunk with finish reason
        yield ChatCompletionChunk(
            id="mock-chunk-id",
            model=self._model_id,
            created=created,
            choices=[
                ChunkChoice(
                    index=0,
                    delta=ChunkDelta(),
                    finish_reason="stop",
                )
            ],
        )

    def tokenize(self, text: str) -> list[int]:
        """Mock tokenization - returns character codes."""
        return [ord(c) for c in text]

    def count_tokens(self, text: str) -> int:
        """Mock token count - returns character count."""
        return len(text)

    async def load(self) -> None:
        """Mock load - sets is_loaded to True."""
        self._is_loaded = True

    async def unload(self) -> None:
        """Mock unload - sets is_loaded to False."""
        self._is_loaded = False
