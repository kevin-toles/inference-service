"""LlamaCpp-based inference provider.

Provides LLM inference using llama-cpp-python with Metal acceleration on Mac.

Patterns applied:
- InferenceProvider ABC implementation
- Async context manager for resource management
- Exception classes ending in "Error" (AP-7)
- No mutable default arguments (AP-1.5)
- PEP 604 union syntax (X | None)

Reference: WBS-INF5 AC-5.1 through AC-5.6
"""

from __future__ import annotations

import asyncio
import platform
import time
from collections.abc import AsyncIterator
from pathlib import Path
from typing import TYPE_CHECKING, Any

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


# Import Llama at module level for easier mocking in tests
# Use try/except to handle case where llama-cpp-python is not installed
try:
    from llama_cpp import Llama
except ImportError:
    Llama = None  # type: ignore[misc, assignment]

if TYPE_CHECKING:
    from llama_cpp import Llama as LlamaType


# =============================================================================
# Constants (S1192: Avoid duplicated string literals)
# =============================================================================

STATUS_AVAILABLE = "available"
STATUS_LOADED = "loaded"
STATUS_LOADING = "loading"
DEFAULT_CONTEXT_LENGTH = 4096
TOKENS_PER_CHAR_ESTIMATE = 4  # Rough estimate: ~4 chars per token


# =============================================================================
# Exceptions (AP-7: Exception names end in "Error")
# =============================================================================


class LlamaCppProviderError(Exception):
    """Base exception for LlamaCpp provider errors.

    All LlamaCpp-specific exceptions inherit from this class.
    """


class LlamaCppModelNotFoundError(LlamaCppProviderError):
    """Raised when model file is not found on disk.

    AC-5.1: Invalid model path raises this error.
    """


class LlamaCppModelLoadError(LlamaCppProviderError):
    """Raised when model fails to load.

    AC-5.6: load() raises this error on failure.
    """


class LlamaCppInferenceError(LlamaCppProviderError):
    """Raised when inference fails.

    AC-5.2: generate() raises this error when model not loaded.
    """


# =============================================================================
# LlamaCppProvider Implementation
# =============================================================================


class LlamaCppProvider(InferenceProvider):
    """LlamaCpp-based inference provider.

    Uses llama-cpp-python library for GGUF model inference with
    Metal acceleration on Mac.

    AC-5.1: Loads GGUF models from disk
    AC-5.2: Generates completions using llama-cpp-python
    AC-5.3: Supports streaming with proper SSE format
    AC-5.4: Correctly reports token usage
    AC-5.5: Uses Metal acceleration on Mac (n_gpu_layers=-1)
    AC-5.6: Handles model loading/unloading with context manager

    Args:
        model_path: Path to the GGUF model file.
        model_id: Unique model identifier.
        context_length: Maximum context window in tokens.
        n_gpu_layers: Number of layers to offload to GPU (-1 for all).
        roles: List of roles this model supports.
        memory_mb: Memory requirement in megabytes.

    Raises:
        LlamaCppModelNotFoundError: If model file does not exist.

    Example:
        >>> provider = LlamaCppProvider(
        ...     model_path=Path("/models/phi-4.gguf"),
        ...     model_id="phi-4",
        ...     context_length=16384,
        ...     n_gpu_layers=-1,  # Metal acceleration
        ... )
        >>> async with provider:
        ...     response = await provider.generate(request)
    """

    def __init__(
        self,
        model_path: Path,
        model_id: str,
        context_length: int = DEFAULT_CONTEXT_LENGTH,
        n_gpu_layers: int = 0,
        roles: list[str] | None = None,
        memory_mb: int = 0,
    ) -> None:
        """Initialize LlamaCpp provider.

        Args:
            model_path: Path to GGUF model file.
            model_id: Unique model identifier.
            context_length: Maximum context length.
            n_gpu_layers: GPU layers (-1 for all, Metal on Mac).
            roles: Model roles (default_factory pattern for AP-1.5).
            memory_mb: Memory requirement in MB.

        Raises:
            LlamaCppModelNotFoundError: If model path doesn't exist.
        """
        self._model_path = Path(model_path)
        self._model_id = model_id
        self._context_length = context_length
        self._n_gpu_layers = n_gpu_layers
        self._roles: list[str] = roles if roles is not None else []
        self._memory_mb = memory_mb

        # Model instance (lazy loaded)
        self._model: LlamaType | None = None
        self._is_loaded = False

        # Inference lock to serialize concurrent requests
        # llama-cpp-python is NOT thread-safe for concurrent llama_decode() calls
        # Reference: Python Cookbook 3rd Ed, Ch12.4 "Locking Critical Sections"
        # WBS-FIX: Prevents SIGABRT from concurrent batch_allocr corruption
        self._inference_lock = asyncio.Lock()

        # Validate model path exists
        if not self._model_path.exists():
            raise LlamaCppModelNotFoundError(
                f"Model file not found: {self._model_path}"
            )

    # =========================================================================
    # Properties
    # =========================================================================

    @property
    def model_info(self) -> ModelMetadata:
        """Get model metadata.

        AC-4.3: Provider exposes model metadata (context length, roles).

        Returns:
            ModelMetadata with model information.
        """
        status = STATUS_LOADED if self._is_loaded else STATUS_AVAILABLE
        return ModelMetadata(
            model_id=self._model_id,
            context_length=self._context_length,
            roles=self._roles.copy(),  # Return copy to prevent mutation
            memory_mb=self._memory_mb,
            status=status,
            file_path=str(self._model_path),
        )

    @property
    def is_loaded(self) -> bool:
        """Check if model is currently loaded.

        Returns:
            True if model is loaded and ready for inference.
        """
        return self._is_loaded

    # =========================================================================
    # Load / Unload
    # =========================================================================

    async def load(self) -> None:
        """Load model into memory.

        AC-5.1: Loads GGUF models from disk.
        AC-5.5: Uses Metal acceleration on Mac with n_gpu_layers=-1.

        Raises:
            LlamaCppModelLoadError: If model fails to load.
        """
        # Idempotent - skip if already loaded
        if self._is_loaded:
            return

        try:
            # Check if llama-cpp-python is available
            if Llama is None:
                raise LlamaCppModelLoadError(
                    "llama-cpp-python is not installed. "
                    "Install with: pip install llama-cpp-python"
                )

            # Determine GPU layers
            # AC-5.5: n_gpu_layers=-1 activates Metal on Mac
            n_gpu_layers = self._n_gpu_layers
            if n_gpu_layers == -1 and platform.system() == "Darwin":
                # Metal acceleration for all layers
                n_gpu_layers = -1

            # Run synchronous model loading in thread pool
            # to avoid blocking the event loop
            self._model = await asyncio.to_thread(
                Llama,
                model_path=str(self._model_path),
                n_ctx=self._context_length,
                n_gpu_layers=n_gpu_layers,
                verbose=False,
            )

            self._is_loaded = True

        except LlamaCppModelLoadError:
            raise
        except Exception as e:
            raise LlamaCppModelLoadError(
                f"Failed to load model {self._model_id}: {e}"
            ) from e

    async def unload(self) -> None:
        """Unload model from memory.

        AC-5.6: Provider handles model loading/unloading.
        """
        if self._model is not None:
            # Release model reference
            self._model = None
        self._is_loaded = False

    # =========================================================================
    # Context Manager (AC-5.6)
    # =========================================================================

    async def __aenter__(self) -> LlamaCppProvider:
        """Enter async context manager - load model.

        AC-5.6: Provider supports context manager for resource management.

        Returns:
            Self with model loaded.
        """
        await self.load()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        """Exit async context manager - unload model.

        Ensures model is unloaded even if exception occurred.

        Args:
            exc_type: Exception type if raised.
            exc_val: Exception value if raised.
            exc_tb: Exception traceback if raised.
        """
        await self.unload()

    # =========================================================================
    # Generation
    # =========================================================================

    async def generate(
        self, request: ChatCompletionRequest
    ) -> ChatCompletionResponse:
        """Generate a non-streaming completion.

        AC-5.2: Provider generates completions using llama-cpp-python.
        AC-5.4: Provider correctly reports token usage.

        Args:
            request: Chat completion request with messages.

        Returns:
            ChatCompletionResponse with choices and usage.

        Raises:
            LlamaCppInferenceError: If model not loaded or generation fails.
        """
        if not self._is_loaded or self._model is None:
            raise LlamaCppInferenceError(
                f"Model {self._model_id} is not loaded. Call load() first."
            )

        try:
            # Acquire inference lock to serialize concurrent requests
            # llama-cpp-python C++ backend crashes with concurrent llama_decode()
            async with self._inference_lock:
                # Convert messages to format expected by llama-cpp-python
                messages = self._convert_messages(request.messages)

                # Build generation kwargs
                gen_kwargs = self._build_generation_kwargs(request)

                # Run synchronous generation in thread pool
                result = await asyncio.to_thread(
                    self._model.create_chat_completion,
                    messages=messages,  # type: ignore[arg-type]
                    **gen_kwargs,
                )

                # Parse response
                return self._parse_completion_response(result)  # type: ignore[arg-type]

        except LlamaCppInferenceError:
            raise
        except Exception as e:
            raise LlamaCppInferenceError(
                f"Generation failed for {self._model_id}: {e}"
            ) from e

    async def stream(  # type: ignore[override]
        self, request: ChatCompletionRequest
    ) -> AsyncIterator[ChatCompletionChunk]:
        """Generate a streaming completion.

        AC-5.3: Provider supports streaming with proper SSE format.

        Args:
            request: Chat completion request with messages.

        Yields:
            ChatCompletionChunk for each token/segment.

        Raises:
            LlamaCppInferenceError: If model not loaded.
        """
        if not self._is_loaded or self._model is None:
            raise LlamaCppInferenceError(
                f"Model {self._model_id} is not loaded. Call load() first."
            )

        try:
            # Acquire inference lock for ENTIRE streaming session
            # llama-cpp-python C++ backend crashes with concurrent llama_decode()
            async with self._inference_lock:
                # Convert messages
                messages = self._convert_messages(request.messages)

                # Build generation kwargs with stream=True
                gen_kwargs = self._build_generation_kwargs(request)
                gen_kwargs["stream"] = True

                # Get streaming iterator (synchronous)
                # Run initial setup in thread pool
                stream_iter = await asyncio.to_thread(
                    self._model.create_chat_completion,
                    messages=messages,  # type: ignore[arg-type]
                    **gen_kwargs,
                )

                # Iterate over chunks (while holding lock)
                for chunk in stream_iter:
                    yield self._parse_streaming_chunk(chunk)  # type: ignore[arg-type]

        except LlamaCppInferenceError:
            raise
        except Exception as e:
            raise LlamaCppInferenceError(
                f"Streaming failed for {self._model_id}: {e}"
            ) from e

    # =========================================================================
    # Tokenization
    # =========================================================================

    def tokenize(self, text: str) -> list[int]:
        """Tokenize text into token IDs.

        AC-5.4: Uses model's tokenizer when loaded.

        Args:
            text: Text string to tokenize.

        Returns:
            List of token IDs.
        """
        if self._model is not None:
            # Use model's tokenizer
            # llama-cpp-python expects bytes
            return self._model.tokenize(text.encode("utf-8"))
        else:
            # Fallback: simple character-based estimate
            # This is a rough approximation
            return list(range(len(text) // TOKENS_PER_CHAR_ESTIMATE))

    def count_tokens(self, text: str) -> int:
        """Count tokens in text.

        AC-5.4: Provider correctly reports token usage.

        Args:
            text: Text string to count tokens in.

        Returns:
            Number of tokens.
        """
        if self._model is not None:
            return len(self.tokenize(text))
        else:
            # Fallback estimate when model not loaded
            # ~4 characters per token is a reasonable heuristic
            return max(1, len(text) // TOKENS_PER_CHAR_ESTIMATE)

    # =========================================================================
    # Helper Methods
    # =========================================================================

    def _convert_messages(
        self, messages: list[Any]
    ) -> list[dict[str, str]]:
        """Convert request messages to llama-cpp format.

        Args:
            messages: List of Message objects.

        Returns:
            List of message dicts for llama-cpp-python.
        """
        converted: list[dict[str, str]] = []
        for msg in messages:
            converted.append({
                "role": msg.role,
                "content": msg.content or "",
            })
        return converted

    def _build_generation_kwargs(
        self, request: ChatCompletionRequest
    ) -> dict[str, Any]:
        """Build generation kwargs from request.

        Args:
            request: Chat completion request.

        Returns:
            Dict of kwargs for create_chat_completion.
        """
        kwargs: dict[str, Any] = {}

        if request.max_tokens is not None:
            kwargs["max_tokens"] = request.max_tokens

        if request.temperature is not None:
            kwargs["temperature"] = request.temperature

        if request.top_p is not None:
            kwargs["top_p"] = request.top_p

        if request.stop is not None:
            kwargs["stop"] = request.stop

        if request.presence_penalty is not None:
            kwargs["presence_penalty"] = request.presence_penalty

        if request.frequency_penalty is not None:
            kwargs["frequency_penalty"] = request.frequency_penalty

        return kwargs

    def _parse_completion_response(
        self, result: dict[str, Any]
    ) -> ChatCompletionResponse:
        """Parse llama-cpp response to ChatCompletionResponse.

        Args:
            result: Raw response from llama-cpp-python.

        Returns:
            Parsed ChatCompletionResponse.
        """
        # Extract choices
        choices: list[Choice] = []
        for i, raw_choice in enumerate(result.get("choices", [])):
            message_data = raw_choice.get("message", {})
            choices.append(
                Choice(
                    index=i,
                    message=ChoiceMessage(
                        role=message_data.get("role", "assistant"),
                        content=message_data.get("content"),
                        tool_calls=message_data.get("tool_calls"),
                    ),
                    finish_reason=raw_choice.get("finish_reason"),
                )
            )

        # Extract usage
        usage_data = result.get("usage", {})
        usage = Usage(
            prompt_tokens=usage_data.get("prompt_tokens", 0),
            completion_tokens=usage_data.get("completion_tokens", 0),
            total_tokens=usage_data.get("total_tokens", 0),
        )

        return ChatCompletionResponse(
            id=result.get("id", f"chatcmpl-{int(time.time())}"),
            model=self._model_id,  # Use our model_id
            created=result.get("created", int(time.time())),
            choices=choices,
            usage=usage,
        )

    def _parse_streaming_chunk(
        self, chunk: dict[str, Any]
    ) -> ChatCompletionChunk:
        """Parse llama-cpp streaming chunk to ChatCompletionChunk.

        Args:
            chunk: Raw streaming chunk from llama-cpp-python.

        Returns:
            Parsed ChatCompletionChunk.
        """
        # Extract chunk choices
        chunk_choices: list[ChunkChoice] = []
        for raw_choice in chunk.get("choices", []):
            delta_data = raw_choice.get("delta", {})
            chunk_choices.append(
                ChunkChoice(
                    index=raw_choice.get("index", 0),
                    delta=ChunkDelta(
                        role=delta_data.get("role"),
                        content=delta_data.get("content"),
                        tool_calls=delta_data.get("tool_calls"),
                    ),
                    finish_reason=raw_choice.get("finish_reason"),
                )
            )

        return ChatCompletionChunk(
            id=chunk.get("id", f"chatcmpl-{int(time.time())}"),
            model=self._model_id,
            created=chunk.get("created", int(time.time())),
            choices=chunk_choices,
        )
