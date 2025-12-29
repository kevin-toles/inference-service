"""Error handlers for FastAPI exception handling.

This module implements:
- AC-18.3: Error responses match llm-gateway schema
- AC-18.4: FastAPI exception handlers registered

Error Response Schema (aligned with llm-gateway):
{
    "error": {
        "code": "ERROR_CODE",
        "message": "Human-readable message",
        "type": "retriable|non_retriable",
        "provider": "inference-service",
        "details": {...}
    }
}

Reference: ARCHITECTURE.md → Error Response Format
"""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from src.core.exceptions import (
    ContextBudgetExceededError,
    ErrorCode,
    InferenceServiceError,
    ModelBusyError,
    ModelLoadingError,
    ModelNotFoundError,
    NonRetriableError,
    QueueFullError,
    RetriableError,
    ValidationError,
)


# =============================================================================
# Error Response Models (Pydantic)
# =============================================================================


class ErrorDetail(BaseModel):
    """Error detail schema matching llm-gateway format.

    Attributes:
        code: Machine-readable error code.
        message: Human-readable error message.
        type: Error type (retriable or non_retriable).
        provider: Service that generated the error.
        details: Additional error-specific information.
    """

    code: str
    message: str
    type: str
    provider: str = "inference-service"
    details: dict[str, Any] | None = None


class ErrorResponse(BaseModel):
    """Error response wrapper matching llm-gateway schema.

    Attributes:
        error: The error detail object.
    """

    error: ErrorDetail


# =============================================================================
# Status Code Mapping
# =============================================================================


def get_status_code_for_error(error: Exception) -> int:
    """Determine HTTP status code based on exception type.

    Args:
        error: The exception to get status code for.

    Returns:
        Appropriate HTTP status code.
    """
    # Specific error types with specific codes
    if isinstance(error, ModelNotFoundError):
        return 404
    if isinstance(error, (ValidationError, ContextBudgetExceededError)):
        return 400
    if isinstance(error, (ModelBusyError, ModelLoadingError, QueueFullError)):
        return 503

    # General categories
    if isinstance(error, RetriableError):
        return 503  # Service Unavailable
    if isinstance(error, NonRetriableError):
        return 500  # Internal Server Error
    if isinstance(error, InferenceServiceError):
        return 500

    # Unknown errors
    return 500


# =============================================================================
# Error Details Extraction
# =============================================================================


def extract_error_details(error: Exception) -> dict[str, Any]:
    """Extract additional details from exception attributes.

    Args:
        error: The exception to extract details from.

    Returns:
        Dictionary of error details.
    """
    details: dict[str, Any] = {}

    # Extract known attributes
    known_attrs = [
        "model_id",
        "model",
        "current_tokens",
        "budget",
        "compression_attempted",
        "max_concurrent",
        "current_size",
        "resource_type",
        "progress",
        "iterations",
        "target_ratio",
        "missing_fields",
        "available_models",
        "mode",
        "completed_steps",
        "total_steps",
        "field",
        "value",
        "setting",
    ]

    for attr in known_attrs:
        if hasattr(error, attr):
            value = getattr(error, attr)
            if value is not None:
                details[attr] = value

    return details if details else {}


# =============================================================================
# Error Response Builder
# =============================================================================


def build_error_response(
    error: Exception,
    error_type: str = "non_retriable",
) -> ErrorResponse:
    """Build a standardized error response.

    Args:
        error: The exception that occurred.
        error_type: Either "retriable" or "non_retriable".

    Returns:
        ErrorResponse with structured error information.
    """
    # Get error code - use ternary for SIM108 compliance
    code = (
        error.error_code
        if hasattr(error, "error_code")
        else ErrorCode.INFERENCE_ERROR.value
    )

    # Get message - use ternary for SIM108 compliance
    message = error.message if hasattr(error, "message") else str(error)

    # Extract details
    details = extract_error_details(error)

    return ErrorResponse(
        error=ErrorDetail(
            code=code,
            message=message,
            type=error_type,
            provider="inference-service",
            details=details if details else None,
        )
    )


# =============================================================================
# Exception Handlers
# =============================================================================


async def inference_service_error_handler(
    _request: Request,
    exc: Exception,
) -> JSONResponse:
    """Handle base InferenceServiceError exceptions.

    Args:
        _request: The FastAPI request (unused).
        exc: The exception that was raised.

    Returns:
        JSONResponse with error details.
    """
    # Cast to InferenceServiceError for type checking
    service_exc = exc if isinstance(exc, InferenceServiceError) else None
    if service_exc is None:
        # Fallback for unexpected cases
        return await generic_error_handler(_request, exc)

    error_type = "non_retriable"
    if isinstance(service_exc, RetriableError):
        error_type = "retriable"

    response = build_error_response(service_exc, error_type)
    status_code = get_status_code_for_error(service_exc)

    return JSONResponse(
        status_code=status_code,
        content=response.model_dump(),
    )


async def retriable_error_handler(
    _request: Request,
    exc: Exception,
) -> JSONResponse:
    """Handle RetriableError exceptions with Retry-After header.

    Args:
        _request: The FastAPI request (unused).
        exc: The retriable exception that was raised.

    Returns:
        JSONResponse with error details and Retry-After header.
    """
    # Cast to RetriableError for type checking
    retriable_exc = exc if isinstance(exc, RetriableError) else None
    if retriable_exc is None:
        # Fallback for unexpected cases
        return await generic_error_handler(_request, exc)

    response = build_error_response(retriable_exc, "retriable")
    status_code = get_status_code_for_error(retriable_exc)

    # Calculate Retry-After in seconds
    retry_after_seconds = retriable_exc.retry_after_ms // 1000
    if retry_after_seconds == 0:
        retry_after_seconds = 1

    return JSONResponse(
        status_code=status_code,
        content=response.model_dump(),
        headers={"Retry-After": str(retry_after_seconds)},
    )


async def non_retriable_error_handler(
    _request: Request,
    exc: Exception,
) -> JSONResponse:
    """Handle NonRetriableError exceptions.

    Args:
        _request: The FastAPI request (unused).
        exc: The non-retriable exception that was raised.

    Returns:
        JSONResponse with error details.
    """
    # Cast to NonRetriableError for type checking
    non_retriable_exc = exc if isinstance(exc, NonRetriableError) else None
    if non_retriable_exc is None:
        # Fallback for unexpected cases
        return await generic_error_handler(_request, exc)

    response = build_error_response(non_retriable_exc, "non_retriable")
    status_code = get_status_code_for_error(non_retriable_exc)

    return JSONResponse(
        status_code=status_code,
        content=response.model_dump(),
    )


async def validation_error_handler(
    _request: Request,
    exc: Exception,
) -> JSONResponse:
    """Handle ValidationError exceptions.

    Args:
        _request: The FastAPI request (unused).
        exc: The validation exception that was raised.

    Returns:
        JSONResponse with 400 status code.
    """
    # Cast to ValidationError for type checking
    validation_exc = exc if isinstance(exc, ValidationError) else None
    if validation_exc is None:
        # Fallback for unexpected cases
        return await generic_error_handler(_request, exc)

    response = build_error_response(validation_exc, "non_retriable")

    return JSONResponse(
        status_code=400,
        content=response.model_dump(),
    )


def generic_error_handler(
    _request: Request,
    exc: Exception,
) -> JSONResponse:
    """Handle generic/unexpected exceptions.

    Args:
        _request: The FastAPI request (unused).
        exc: The exception that was raised.

    Returns:
        JSONResponse with 500 status code.
    """
    response = ErrorResponse(
        error=ErrorDetail(
            code=ErrorCode.INFERENCE_ERROR.value,
            message=f"Internal server error: {exc!s}",
            type="non_retriable",
            provider="inference-service",
            details=None,
        )
    )

    return JSONResponse(
        status_code=500,
        content=response.model_dump(),
    )


# =============================================================================
# Handler Registration
# =============================================================================


def register_exception_handlers(app: FastAPI) -> None:
    """Register all exception handlers with the FastAPI application.

    This function should be called during application startup to
    register all custom exception handlers.

    Args:
        app: The FastAPI application instance.
    """
    # Register from most specific to least specific
    # More specific handlers should be registered first

    # Specific retriable errors
    app.add_exception_handler(ModelBusyError, retriable_error_handler)
    app.add_exception_handler(ModelLoadingError, retriable_error_handler)
    app.add_exception_handler(QueueFullError, retriable_error_handler)

    # Specific non-retriable errors
    app.add_exception_handler(ModelNotFoundError, non_retriable_error_handler)
    app.add_exception_handler(ValidationError, validation_error_handler)
    app.add_exception_handler(ContextBudgetExceededError, non_retriable_error_handler)

    # Base exception handlers
    app.add_exception_handler(RetriableError, retriable_error_handler)
    app.add_exception_handler(NonRetriableError, non_retriable_error_handler)
    app.add_exception_handler(InferenceServiceError, inference_service_error_handler)

    # Catch-all for unexpected exceptions
    app.add_exception_handler(Exception, generic_error_handler)
