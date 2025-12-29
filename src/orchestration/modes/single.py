"""Single mode orchestration - direct pass-through to one model.

SingleMode implements the simplest orchestration: Request → Model → Response.
No critique, no multi-model coordination - just direct inference.

Flow:
    Request → Model → Response

Per ARCHITECTURE.md Orchestration Modes:
- single: One model, no critique
- Min Models: 1
- Flow: Request → Model → Response

Reference: WBS-INF11 AC-11.1, AC-11.2, AC-11.3
"""

import time
from collections.abc import AsyncIterator

from src.models.requests import ChatCompletionRequest
from src.models.responses import (
    ChatCompletionChunk,
    ChatCompletionResponse,
    OrchestrationMetadata,
)
from src.providers.base import InferenceProvider


class SingleMode:
    """Single model orchestration mode.

    Passes requests directly to one model without any multi-model
    orchestration. This is the simplest mode - just a thin wrapper
    that adds orchestration metadata to the response.

    AC-11.1: Passes request directly to one model.
    AC-11.2: Returns response with usage stats.
    AC-11.3: Supports streaming.

    Attributes:
        provider: The inference provider to use for generation.
        model_id: Model identifier from the provider.

    Example:
        provider = LlamaCppProvider(model_path="phi-4.gguf")
        mode = SingleMode(provider=provider)
        response = await mode.execute(request)
    """

    def __init__(self, provider: InferenceProvider) -> None:
        """Initialize SingleMode with an inference provider.

        Args:
            provider: Inference provider for model execution.
        """
        self._provider = provider

    @property
    def provider(self) -> InferenceProvider:
        """Get the inference provider."""
        return self._provider

    @property
    def model_id(self) -> str:
        """Get the model ID from the provider."""
        return self._provider.model_info.model_id

    async def execute(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        """Execute a single model completion.

        AC-11.1: Passes request directly to provider.generate().
        AC-11.2: Returns response with usage stats and orchestration metadata.

        Args:
            request: Chat completion request.

        Returns:
            ChatCompletionResponse with orchestration metadata added.

        Raises:
            Any errors from the provider are propagated.
        """
        start_time = time.perf_counter()

        # Pass request directly to provider (AC-11.1)
        response = await self._provider.generate(request)

        # Calculate inference time
        inference_time_ms = (time.perf_counter() - start_time) * 1000

        # Add orchestration metadata (AC-11.2)
        response.orchestration = OrchestrationMetadata(
            mode="single",
            models_used=[self.model_id],
            total_inference_time_ms=inference_time_ms,
            rounds=None,  # Not applicable for single mode
            final_score=None,  # Not applicable for single mode
            agreement_score=None,  # Not applicable for single mode
        )

        return response

    async def stream(
        self, request: ChatCompletionRequest
    ) -> AsyncIterator[ChatCompletionChunk]:
        """Stream a single model completion.

        AC-11.3: Supports streaming through provider.stream().

        Yields chunks directly from the provider without modification.
        Orchestration metadata is not included in streaming responses
        (chunks don't have orchestration field per OpenAI spec).

        Args:
            request: Chat completion request.

        Yields:
            ChatCompletionChunk for each generated token/segment.

        Raises:
            Any errors from the provider are propagated.
        """
        # Provider.stream() is an async generator, iterate directly
        # Type ignore needed because mypy sees async def -> AsyncIterator
        # as coroutine, but it's actually an async generator
        async for chunk in self._provider.stream(request):  # type: ignore[attr-defined]
            yield chunk
