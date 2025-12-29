"""Custom exceptions for inference-service.

This module implements the exception hierarchy for the inference-service:
- AC-18.1: Exception hierarchy with Retriable/NonRetriable
- AC-18.2: All custom exceptions end in "Error" (AP-7)

Exception Hierarchy:
    InferenceServiceError (base)
    ├── RetriableError (transient errors)
    │   ├── ModelBusyError
    │   ├── ModelLoadingError
    │   ├── TemporaryResourceError
    │   └── QueueFullError
    └── NonRetriableError (permanent errors)
        ├── ContextBudgetExceededError
        ├── CompressionFailedError
        ├── HandoffStateInvalidError
        ├── ModelNotFoundError
        ├── OrchestrationFailedError
        ├── ValidationError
        └── ConfigurationError

Reference: ARCHITECTURE.md → Error Handling
"""

from __future__ import annotations

from enum import Enum
from typing import Any


# =============================================================================
# Error Codes Enum
# =============================================================================


class ErrorCode(str, Enum):
    """Error codes for inference-service exceptions.

    These codes provide a consistent way to identify error types
    across the API and in logging.
    """

    # Base error
    INFERENCE_ERROR = "INFERENCE_ERROR"

    # Retriable errors
    MODEL_BUSY = "MODEL_BUSY"
    MODEL_LOADING = "MODEL_LOADING"
    TEMPORARY_RESOURCE = "TEMPORARY_RESOURCE"
    QUEUE_FULL = "QUEUE_FULL"

    # Non-retriable errors
    CONTEXT_BUDGET_EXCEEDED = "CONTEXT_BUDGET_EXCEEDED"
    COMPRESSION_FAILED = "COMPRESSION_FAILED"
    HANDOFF_STATE_INVALID = "HANDOFF_STATE_INVALID"
    MODEL_NOT_FOUND = "MODEL_NOT_FOUND"
    ORCHESTRATION_FAILED = "ORCHESTRATION_FAILED"
    VALIDATION_ERROR = "VALIDATION_ERROR"
    CONFIGURATION_ERROR = "CONFIGURATION_ERROR"


# =============================================================================
# Base Exception
# =============================================================================


class InferenceServiceError(Exception):
    """Base exception for all inference-service errors.

    All custom exceptions inherit from this class, providing
    consistent error handling and structured error information.

    Attributes:
        message: Human-readable error message.
        error_code: Machine-readable error code from ErrorCode enum.
    """

    def __init__(
        self,
        message: str,
        error_code: str | ErrorCode = ErrorCode.INFERENCE_ERROR,
        **kwargs: Any,
    ) -> None:
        """Initialize the exception.

        Args:
            message: Human-readable error message.
            error_code: Machine-readable error code.
            **kwargs: Additional attributes to set on the exception.
        """
        super().__init__(message)
        self.message = message
        self.error_code = (
            error_code.value if isinstance(error_code, ErrorCode) else error_code
        )

        # Set any additional attributes from kwargs
        for key, value in kwargs.items():
            setattr(self, key, value)


# =============================================================================
# Retriable Error Base
# =============================================================================


class RetriableError(InferenceServiceError):
    """Base class for transient errors that may succeed on retry.

    Retriable errors indicate temporary conditions that may resolve
    if the request is retried after a delay.

    Attributes:
        retry_after_ms: Suggested retry delay in milliseconds.
    """

    def __init__(
        self,
        message: str,
        retry_after_ms: int = 1000,
        error_code: str | ErrorCode = ErrorCode.INFERENCE_ERROR,
        **kwargs: Any,
    ) -> None:
        """Initialize the retriable error.

        Args:
            message: Human-readable error message.
            retry_after_ms: Suggested retry delay in milliseconds (default: 1000).
            error_code: Machine-readable error code.
            **kwargs: Additional attributes.
        """
        super().__init__(message, error_code, **kwargs)
        self.retry_after_ms = retry_after_ms


# =============================================================================
# Non-Retriable Error Base
# =============================================================================


class NonRetriableError(InferenceServiceError):
    """Base class for permanent errors that should not be retried.

    Non-retriable errors indicate conditions that will not change
    regardless of retry attempts.
    """

    pass


# =============================================================================
# Retriable Exceptions
# =============================================================================


class ModelBusyError(RetriableError):
    """Model is currently processing another request.

    This error indicates the model is busy and the request should
    be retried after a short delay.

    Attributes:
        model_id: ID of the busy model.
    """

    def __init__(
        self,
        message: str,
        model_id: str | None = None,
        retry_after_ms: int = 1000,
        **kwargs: Any,
    ) -> None:
        """Initialize ModelBusyError.

        Args:
            message: Error message.
            model_id: ID of the busy model.
            retry_after_ms: Suggested retry delay.
            **kwargs: Additional attributes.
        """
        super().__init__(
            message,
            retry_after_ms=retry_after_ms,
            error_code=ErrorCode.MODEL_BUSY,
            **kwargs,
        )
        self.model_id = model_id


class ModelLoadingError(RetriableError):
    """Model is still loading into memory.

    This error indicates the model is being loaded and the request
    should be retried after the loading completes.

    Attributes:
        model_id: ID of the loading model.
        progress: Loading progress (0.0 to 1.0).
    """

    def __init__(
        self,
        message: str,
        model_id: str | None = None,
        progress: float | None = None,
        retry_after_ms: int = 2000,
        **kwargs: Any,
    ) -> None:
        """Initialize ModelLoadingError.

        Args:
            message: Error message.
            model_id: ID of the loading model.
            progress: Loading progress (0.0 to 1.0).
            retry_after_ms: Suggested retry delay.
            **kwargs: Additional attributes.
        """
        super().__init__(
            message,
            retry_after_ms=retry_after_ms,
            error_code=ErrorCode.MODEL_LOADING,
            **kwargs,
        )
        self.model_id = model_id
        self.progress = progress


class TemporaryResourceError(RetriableError):
    """Temporary resource exhaustion (e.g., memory pressure).

    This error indicates temporary resource constraints that may
    resolve if the request is retried later.

    Attributes:
        resource_type: Type of resource that is exhausted.
    """

    def __init__(
        self,
        message: str,
        resource_type: str | None = None,
        retry_after_ms: int = 5000,
        **kwargs: Any,
    ) -> None:
        """Initialize TemporaryResourceError.

        Args:
            message: Error message.
            resource_type: Type of exhausted resource (e.g., "memory", "gpu").
            retry_after_ms: Suggested retry delay.
            **kwargs: Additional attributes.
        """
        super().__init__(
            message,
            retry_after_ms=retry_after_ms,
            error_code=ErrorCode.TEMPORARY_RESOURCE,
            **kwargs,
        )
        self.resource_type = resource_type


class QueueFullError(RetriableError):
    """Request queue is at capacity.

    This error indicates the request queue is full and the request
    should be retried after some requests complete.

    Attributes:
        max_concurrent: Maximum concurrent requests allowed.
        current_size: Current number of requests in queue.
    """

    def __init__(
        self,
        message: str,
        max_concurrent: int | None = None,
        current_size: int | None = None,
        retry_after_ms: int = 1000,
        **kwargs: Any,
    ) -> None:
        """Initialize QueueFullError.

        Args:
            message: Error message.
            max_concurrent: Maximum concurrent requests.
            current_size: Current queue size.
            retry_after_ms: Suggested retry delay.
            **kwargs: Additional attributes.
        """
        super().__init__(
            message,
            retry_after_ms=retry_after_ms,
            error_code=ErrorCode.QUEUE_FULL,
            **kwargs,
        )
        self.max_concurrent = max_concurrent
        self.current_size = current_size


# =============================================================================
# Non-Retriable Exceptions
# =============================================================================


class ContextBudgetExceededError(NonRetriableError):
    """Content cannot fit in context window even after compression.

    This error indicates the input content exceeds the model's context
    budget and cannot be compressed further.

    Attributes:
        current_tokens: Current token count.
        budget: Maximum token budget.
        model: Model that has the budget constraint.
    """

    def __init__(
        self,
        current_tokens: int,
        budget: int,
        model: str | None = None,
        compression_attempted: bool = True,
        **kwargs: Any,
    ) -> None:
        """Initialize ContextBudgetExceededError.

        Args:
            current_tokens: Current token count.
            budget: Maximum token budget.
            model: Model with the budget constraint.
            compression_attempted: Whether compression was attempted.
            **kwargs: Additional attributes.
        """
        message = f"Context budget exceeded: {current_tokens}/{budget} tokens"
        if model:
            message = f"Content cannot fit in {model} context window ({current_tokens}/{budget} tokens)"

        super().__init__(
            message,
            error_code=ErrorCode.CONTEXT_BUDGET_EXCEEDED,
            **kwargs,
        )
        self.current_tokens = current_tokens
        self.budget = budget
        self.model = model
        self.compression_attempted = compression_attempted


class CompressionFailedError(NonRetriableError):
    """Compression could not achieve target ratio after max iterations.

    This error indicates the content compression process failed
    to reach the required compression ratio.

    Attributes:
        iterations: Number of compression iterations attempted.
        target_ratio: Target compression ratio.
    """

    def __init__(
        self,
        message: str,
        iterations: int | None = None,
        target_ratio: float | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize CompressionFailedError.

        Args:
            message: Error message.
            iterations: Number of compression iterations.
            target_ratio: Target compression ratio.
            **kwargs: Additional attributes.
        """
        super().__init__(
            message,
            error_code=ErrorCode.COMPRESSION_FAILED,
            **kwargs,
        )
        self.iterations = iterations
        self.target_ratio = target_ratio


class HandoffStateInvalidError(NonRetriableError):
    """HandoffState missing required fields for pipeline step.

    This error indicates the pipeline state is invalid and cannot
    proceed to the next step.

    Attributes:
        missing_fields: List of missing required fields.
    """

    def __init__(
        self,
        message: str,
        missing_fields: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize HandoffStateInvalidError.

        Args:
            message: Error message.
            missing_fields: List of missing required fields.
            **kwargs: Additional attributes.
        """
        super().__init__(
            message,
            error_code=ErrorCode.HANDOFF_STATE_INVALID,
            **kwargs,
        )
        self.missing_fields = missing_fields


class ModelNotFoundError(NonRetriableError):
    """Requested model not available in configuration.

    This error indicates the requested model is not configured
    or available in the service.

    Attributes:
        model_id: ID of the requested model.
        available_models: List of available model IDs.
    """

    def __init__(
        self,
        message: str,
        model_id: str | None = None,
        available_models: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize ModelNotFoundError.

        Args:
            message: Error message.
            model_id: ID of the missing model.
            available_models: List of available models.
            **kwargs: Additional attributes.
        """
        super().__init__(
            message,
            error_code=ErrorCode.MODEL_NOT_FOUND,
            **kwargs,
        )
        self.model_id = model_id
        self.available_models = available_models


class OrchestrationFailedError(NonRetriableError):
    """Multi-model orchestration failed.

    This error indicates the orchestration process (critique,
    debate, ensemble, pipeline) could not complete successfully.

    Attributes:
        mode: Orchestration mode that failed.
        completed_steps: Number of completed steps.
        total_steps: Total number of steps.
    """

    def __init__(
        self,
        message: str,
        mode: str | None = None,
        completed_steps: int | None = None,
        total_steps: int | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize OrchestrationFailedError.

        Args:
            message: Error message.
            mode: Orchestration mode (single, critique, etc.).
            completed_steps: Number of completed steps.
            total_steps: Total number of steps.
            **kwargs: Additional attributes.
        """
        super().__init__(
            message,
            error_code=ErrorCode.ORCHESTRATION_FAILED,
            **kwargs,
        )
        self.mode = mode
        self.completed_steps = completed_steps
        self.total_steps = total_steps


class ValidationError(NonRetriableError):
    """Request validation failed.

    This error indicates the request failed validation beyond
    Pydantic's built-in validation.

    Attributes:
        field: Name of the invalid field.
        value: The invalid value.
    """

    def __init__(
        self,
        message: str,
        field: str | None = None,
        value: Any = None,
        **kwargs: Any,
    ) -> None:
        """Initialize ValidationError.

        Args:
            message: Error message.
            field: Name of the invalid field.
            value: The invalid value.
            **kwargs: Additional attributes.
        """
        super().__init__(
            message,
            error_code=ErrorCode.VALIDATION_ERROR,
            **kwargs,
        )
        self.field = field
        self.value = value


class ConfigurationError(NonRetriableError):
    """Service configuration is invalid.

    This error indicates a configuration problem that prevents
    the service from operating correctly.

    Attributes:
        setting: Name of the problematic setting.
    """

    def __init__(
        self,
        message: str,
        setting: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize ConfigurationError.

        Args:
            message: Error message.
            setting: Name of the problematic setting.
            **kwargs: Additional attributes.
        """
        super().__init__(
            message,
            error_code=ErrorCode.CONFIGURATION_ERROR,
            **kwargs,
        )
        self.setting = setting
