"""Completions API routes for chat completions.

Provides OpenAI-compatible /v1/chat/completions endpoint.

Reference: WBS-INF9 AC-9.1 through AC-9.5
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING

from fastapi import APIRouter, HTTPException, Request, status
from fastapi.responses import StreamingResponse

from src.models.requests import ChatCompletionRequest
from src.models.responses import ChatCompletionResponse


if TYPE_CHECKING:
    from src.services.model_manager import ModelManager

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


async def _stream_response(
    request: Request,
    completion_request: ChatCompletionRequest,
) -> AsyncIterator[str]:
    """Generate SSE stream for chat completion.

    AC-9.3: Streaming response uses SSE format.

    Args:
        request: FastAPI request object
        completion_request: Chat completion request

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

    try:
        async for chunk in provider.stream(completion_request):
            yield chunk.to_sse() + "\n\n"

        # End with [DONE] marker
        yield SSE_DONE
    except Exception as e:
        logger.exception("Streaming error: %s", str(e))
        # For streaming errors, we can't change status code mid-stream
        # Just end the stream
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
        503: {"description": "Service unavailable"},
    },
)
async def create_chat_completion(
    request: Request,
    completion_request: ChatCompletionRequest,
) -> ChatCompletionResponse | StreamingResponse:
    """Create a chat completion.

    AC-9.1: POST /v1/chat/completions accepts OpenAI-compatible request.
    AC-9.2: Non-streaming response matches OpenAI format.
    AC-9.3: Streaming response uses SSE format.
    AC-9.4: Response includes usage statistics.
    AC-9.5: Response includes orchestration metadata.

    Args:
        request: FastAPI request object
        completion_request: Chat completion request

    Returns:
        ChatCompletionResponse for non-streaming, StreamingResponse for streaming

    Raises:
        HTTPException: Various error codes for different failure modes
    """
    # Handle streaming requests
    if completion_request.stream:
        return StreamingResponse(
            _stream_response(request, completion_request),
            media_type=SSE_CONTENT_TYPE,
        )

    # Non-streaming flow
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
