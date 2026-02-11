"""Completions API routes for chat completions.

Provides OpenAI-compatible /v1/chat/completions endpoint.

Reference: WBS-INF9 AC-9.1 through AC-9.5
H-1: Queue-gated inference — acquire/release slot around both paths.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING

from fastapi import APIRouter, HTTPException, Request, status
from fastapi.responses import JSONResponse, StreamingResponse

from src.models.requests import ChatCompletionRequest
from src.models.responses import ChatCompletionResponse
from src.services.queue_manager import QueueFullError


if TYPE_CHECKING:
    from src.services.model_manager import ModelManager
    from src.services.queue_manager import QueueManager

# =============================================================================
# Router Configuration
# =============================================================================

router = APIRouter(tags=["completions"])
logger = logging.getLogger(__name__)

# =============================================================================
# Constants
# =============================================================================

SSE_CONTENT_TYPE = "text/event-stream"
SSE_DONE = "data: [DONE]\n\n"


# =============================================================================
# Helper Functions
# =============================================================================


def _get_model_manager(request: Request) -> ModelManager:
    """Get model manager from app state or raise 503.

    Args:
        request: FastAPI request object

    Returns:
        ModelManager instance

    Raises:
        HTTPException: 503 if model manager not initialized
    """
    manager: ModelManager | None = getattr(
        request.app.state, "model_manager", None
    )
    if manager is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model manager not initialized",
        )
    return manager


def _get_queue_manager(request: Request) -> QueueManager | None:
    """Get queue manager from app state (optional).

    Returns None if QueueManager is not wired into the app, allowing
    backward-compatible operation without queuing.

    Args:
        request: FastAPI request object

    Returns:
        QueueManager instance or None
    """
    return getattr(request.app.state, "queue_manager", None)


async def _stream_response(
    request: Request,
    completion_request: ChatCompletionRequest,
    queue_manager: QueueManager | None = None,
) -> AsyncIterator[str]:
    """Generate SSE stream for chat completion.

    AC-9.3: Streaming response uses SSE format.
    H-1: Holds queue slot for the entire duration of the stream.

    Args:
        request: FastAPI request object
        completion_request: Chat completion request
        queue_manager: Optional QueueManager for concurrency control

    Yields:
        SSE-formatted strings
    """
    manager = _get_model_manager(request)

    try:
        provider = manager.get_provider(completion_request.model)
    except Exception as e:
        # Re-raise as HTTP exception for error handling
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        ) from e

    # H-1: If queue manager present, hold slot for entire stream
    if queue_manager is not None:
        async with queue_manager.acquire():
            try:
                async for chunk in provider.stream(completion_request):
                    yield chunk.to_sse() + "\n\n"
                yield SSE_DONE
            except Exception as e:
                logger.exception("Streaming error: %s", str(e))
                yield SSE_DONE
    else:
        # No queue — original unguarded path
        try:
            async for chunk in provider.stream(completion_request):
                yield chunk.to_sse() + "\n\n"
            yield SSE_DONE
        except Exception as e:
            logger.exception("Streaming error: %s", str(e))
            yield SSE_DONE


# =============================================================================
# Endpoints
# =============================================================================


@router.post(
    "/chat/completions",
    response_model=ChatCompletionResponse,
    summary="Create chat completion",
    description="Creates a model response for the given chat conversation.",
    responses={
        404: {"description": "Model not found or not loaded"},
        422: {"description": "Validation error"},
        500: {"description": "Inference error"},
        503: {"description": "Service unavailable or queue full"},
    },
)
async def create_chat_completion(
    request: Request,
    completion_request: ChatCompletionRequest,
) -> ChatCompletionResponse | StreamingResponse | JSONResponse:
    """Create a chat completion.

    AC-9.1: POST /v1/chat/completions accepts OpenAI-compatible request.
    AC-9.2: Non-streaming response matches OpenAI format.
    AC-9.3: Streaming response uses SSE format.
    AC-9.4: Response includes usage statistics.
    AC-9.5: Response includes orchestration metadata.
    H-1: All inference gated through QueueManager.

    Args:
        request: FastAPI request object
        completion_request: Chat completion request

    Returns:
        ChatCompletionResponse for non-streaming, StreamingResponse for streaming

    Raises:
        HTTPException: Various error codes for different failure modes
    """
    # H-1: Get queue manager (optional — backward compatible)
    queue_manager = _get_queue_manager(request)

    # Handle streaming requests
    if completion_request.stream:
        # H-1: For streaming, the slot is acquired inside the generator
        # so it stays held for the full duration of the SSE stream.
        # QueueFullError can't happen here for FIFO (blocks instead).
        headers = {}
        if queue_manager is not None and queue_manager.is_full:
            headers["X-Queue-Position"] = str(queue_manager.pending_count + 1)
        return StreamingResponse(
            _stream_response(request, completion_request, queue_manager),
            media_type=SSE_CONTENT_TYPE,
            headers=headers if headers else None,
        )

    # Non-streaming flow — gate through queue
    try:
        if queue_manager is not None:
            # H-1: Report queue position if slot isn't immediately available
            response_headers: dict[str, str] = {}
            if queue_manager.is_full:
                response_headers["X-Queue-Position"] = str(
                    queue_manager.pending_count + 1
                )

            async with queue_manager.acquire():
                result = await _execute_non_streaming(request, completion_request)

            # Attach queue headers to the response if they were set
            if response_headers:
                json_response = JSONResponse(
                    content=result.model_dump(),
                    headers=response_headers,
                )
                return json_response
            return result
        else:
            # No queue manager — original unguarded path
            return await _execute_non_streaming(request, completion_request)

    except QueueFullError as e:
        # H-1: Queue full → HTTP 503 with Retry-After header
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "detail": "Inference queue is full. Please retry later.",
                "max_concurrent": e.max_concurrent,
                "retry_after_seconds": 5,
            },
            headers={"Retry-After": "5"},
        )


async def _execute_non_streaming(
    request: Request,
    completion_request: ChatCompletionRequest,
) -> ChatCompletionResponse:
    """Execute a non-streaming inference request.

    Extracted to allow queue wrapping without deep nesting.

    Args:
        request: FastAPI request object
        completion_request: Chat completion request

    Returns:
        ChatCompletionResponse

    Raises:
        HTTPException: 404 if model not found, 500 on inference error
    """
    manager = _get_model_manager(request)

    # Get the provider for the requested model
    try:
        provider = manager.get_provider(completion_request.model)
    except Exception as e:
        # Import here to avoid circular imports
        from src.services.model_manager import ModelNotLoadedError

        if isinstance(e, ModelNotLoadedError):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=str(e),
            ) from e
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting model provider: {e}",
        ) from e

    # Generate completion
    try:
        response: ChatCompletionResponse = await provider.generate(completion_request)
        return response
    except Exception as e:
        logger.exception("Inference error: %s", str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference error: {e}",
        ) from e
