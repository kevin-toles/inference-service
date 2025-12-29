"""Orchestrator - multi-model orchestration dispatcher.

The Orchestrator is the main entry point for orchestrated inference.
It dispatches requests to the appropriate orchestration mode based
on configuration.

Modes per ARCHITECTURE.md:
- single: One model, no critique
- critique: Generate then critique (not yet implemented)
- debate: Parallel then reconcile (not yet implemented)
- ensemble: All vote, synthesize (not yet implemented)
- pipeline: Sequential stages (not yet implemented)

Reference: WBS-INF11 AC-11.4
"""

from collections.abc import AsyncIterator
from enum import Enum
from typing import ClassVar

from src.models.requests import ChatCompletionRequest
from src.models.responses import ChatCompletionChunk, ChatCompletionResponse
from src.orchestration.modes.single import SingleMode
from src.providers.base import InferenceProvider


class OrchestrationMode(Enum):
    """Available orchestration modes.

    Per ARCHITECTURE.md Orchestration Modes:
    - single: One model, no critique
    - critique: Generate then critique
    - debate: Parallel then reconcile
    - ensemble: All vote, synthesize
    - pipeline: Sequential stages

    AC-11.7: OrchestrationMode enum implemented.
    """

    SINGLE = "single"
    CRITIQUE = "critique"
    DEBATE = "debate"
    ENSEMBLE = "ensemble"
    PIPELINE = "pipeline"


class UnsupportedModeError(Exception):
    """Raised when an orchestration mode is not yet implemented.

    Per CODING_PATTERNS_ANALYSIS AP-7: Exception classes should end in "Error".
    """

    def __init__(self, mode: str) -> None:
        """Initialize with the unsupported mode name.

        Args:
            mode: The orchestration mode that is not supported.
        """
        self.mode = mode
        super().__init__(
            f"Orchestration mode '{mode}' is not yet implemented. "
            f"Currently only 'single' mode is supported."
        )


class Orchestrator:
    """Main orchestration dispatcher.

    Routes requests to the appropriate orchestration mode based on
    configuration. Currently only SingleMode is implemented.

    AC-11.4: Orchestrator dispatches to correct mode.

    Attributes:
        provider: The inference provider for model execution.
        mode: The orchestration mode to use.

    Example:
        provider = LlamaCppProvider(model_path="phi-4.gguf")
        orchestrator = Orchestrator(provider=provider, mode="single")
        response = await orchestrator.execute(request)
    """

    # Supported modes (only single for now)
    _SUPPORTED_MODES: ClassVar[set[OrchestrationMode]] = {OrchestrationMode.SINGLE}

    def __init__(
        self,
        provider: InferenceProvider,
        mode: str | OrchestrationMode = OrchestrationMode.SINGLE,
    ) -> None:
        """Initialize Orchestrator with provider and mode.

        Args:
            provider: Inference provider for model execution.
            mode: Orchestration mode (string or enum). Defaults to single.

        Raises:
            ValueError: If mode string is not a valid OrchestrationMode.
        """
        self._provider = provider

        # Convert string to enum if needed
        if isinstance(mode, str):
            self._mode = OrchestrationMode(mode)
        else:
            self._mode = mode

    @property
    def provider(self) -> InferenceProvider:
        """Get the inference provider."""
        return self._provider

    @property
    def mode(self) -> OrchestrationMode:
        """Get the orchestration mode."""
        return self._mode

    async def execute(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        """Execute orchestrated inference.

        Dispatches to the appropriate orchestration mode based on
        configuration.

        AC-11.4: Dispatches to correct mode.

        Args:
            request: Chat completion request.

        Returns:
            ChatCompletionResponse with orchestration metadata.

        Raises:
            UnsupportedModeError: If mode is not yet implemented.
            Any errors from the underlying mode/provider.
        """
        if self._mode == OrchestrationMode.SINGLE:
            single_mode = SingleMode(provider=self._provider)
            return await single_mode.execute(request)

        # Other modes not yet implemented
        raise UnsupportedModeError(self._mode.value)

    async def stream(
        self, request: ChatCompletionRequest
    ) -> AsyncIterator[ChatCompletionChunk]:
        """Stream orchestrated inference.

        Dispatches to the appropriate orchestration mode for streaming.

        Args:
            request: Chat completion request.

        Yields:
            ChatCompletionChunk for each generated token/segment.

        Raises:
            UnsupportedModeError: If mode is not yet implemented.
            Any errors from the underlying mode/provider.
        """
        if self._mode == OrchestrationMode.SINGLE:
            single_mode = SingleMode(provider=self._provider)
            async for chunk in single_mode.stream(request):
                yield chunk
            return

        # Other modes not yet implemented
        raise UnsupportedModeError(self._mode.value)
