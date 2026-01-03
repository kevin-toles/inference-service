"""Unit tests for exception hierarchy.

Tests cover:
- AC-18.1: Exception hierarchy with Retriable/NonRetriable
- AC-18.2: All custom exceptions end in "Error" (AP-7)

Exit Criteria:
- All exception names match regex `.*Error$`
- Exception hierarchy properly organized
"""

import re

import pytest

from src.core.exceptions import (
    # Base exceptions
    InferenceServiceError,
    RetriableError,
    NonRetriableError,
    # Retriable exceptions
    ModelBusyError,
    ModelLoadingError,
    TemporaryResourceError,
    QueueFullError,
    # Non-retriable exceptions
    ContextBudgetExceededError,
    CompressionFailedError,
    HandoffStateInvalidError,
    ModelNotFoundError,
    OrchestrationFailedError,
    ValidationError,
    ConfigurationError,
    # Error codes
    ErrorCode,
)


# =============================================================================
# AC-18.1: Exception Hierarchy Tests
# =============================================================================


class TestInferenceServiceErrorBase:
    """Test the base InferenceServiceError exception."""

    def test_is_exception(self) -> None:
        """InferenceServiceError should be an Exception subclass."""
        assert issubclass(InferenceServiceError, Exception)

    def test_can_be_instantiated(self) -> None:
        """InferenceServiceError should be instantiable with message."""
        error = InferenceServiceError("test error")
        assert str(error) == "test error"

    def test_has_message_attribute(self) -> None:
        """InferenceServiceError should have message attribute."""
        error = InferenceServiceError("test message")
        assert error.message == "test message"

    def test_has_error_code_attribute(self) -> None:
        """InferenceServiceError should have error_code attribute."""
        error = InferenceServiceError("test", error_code=ErrorCode.INFERENCE_ERROR)
        assert error.error_code == ErrorCode.INFERENCE_ERROR

    def test_default_error_code(self) -> None:
        """InferenceServiceError should have default error code."""
        error = InferenceServiceError("test")
        assert error.error_code == ErrorCode.INFERENCE_ERROR


class TestRetriableError:
    """Test RetriableError base class for transient errors."""

    def test_inherits_from_inference_service_error(self) -> None:
        """RetriableError should inherit from InferenceServiceError."""
        assert issubclass(RetriableError, InferenceServiceError)

    def test_has_retry_after_ms(self) -> None:
        """RetriableError should have retry_after_ms attribute."""
        error = RetriableError("transient error", retry_after_ms=2000)
        assert error.retry_after_ms == 2000

    def test_default_retry_after_ms(self) -> None:
        """RetriableError should have default retry_after_ms of 1000."""
        error = RetriableError("transient error")
        assert error.retry_after_ms == 1000

    def test_message_preserved(self) -> None:
        """RetriableError should preserve error message."""
        error = RetriableError("retry me")
        assert str(error) == "retry me"
        assert error.message == "retry me"


class TestNonRetriableError:
    """Test NonRetriableError base class for permanent errors."""

    def test_inherits_from_inference_service_error(self) -> None:
        """NonRetriableError should inherit from InferenceServiceError."""
        assert issubclass(NonRetriableError, InferenceServiceError)

    def test_message_preserved(self) -> None:
        """NonRetriableError should preserve error message."""
        error = NonRetriableError("permanent error")
        assert str(error) == "permanent error"
        assert error.message == "permanent error"


# =============================================================================
# Retriable Exception Tests
# =============================================================================


class TestModelBusyError:
    """Test ModelBusyError for concurrent access issues."""

    def test_is_retriable(self) -> None:
        """ModelBusyError should be a RetriableError."""
        assert issubclass(ModelBusyError, RetriableError)

    def test_instantiation(self) -> None:
        """ModelBusyError should be instantiable."""
        error = ModelBusyError("phi-4 is busy")
        assert "phi-4" in str(error)

    def test_has_model_id_attribute(self) -> None:
        """ModelBusyError should have model_id attribute."""
        error = ModelBusyError("busy", model_id="phi-4")
        assert error.model_id == "phi-4"


class TestModelLoadingError:
    """Test ModelLoadingError for models still loading."""

    def test_is_retriable(self) -> None:
        """ModelLoadingError should be a RetriableError."""
        assert issubclass(ModelLoadingError, RetriableError)

    def test_instantiation(self) -> None:
        """ModelLoadingError should be instantiable."""
        error = ModelLoadingError("Model still loading")
        assert "loading" in str(error).lower()

    def test_has_model_id_attribute(self) -> None:
        """ModelLoadingError should have model_id attribute."""
        error = ModelLoadingError("loading", model_id="deepseek-r1-7b")
        assert error.model_id == "deepseek-r1-7b"

    def test_has_progress_attribute(self) -> None:
        """ModelLoadingError should support progress attribute."""
        error = ModelLoadingError("loading", model_id="phi-4", progress=0.75)
        assert error.progress == pytest.approx(0.75)


class TestTemporaryResourceError:
    """Test TemporaryResourceError for resource exhaustion."""

    def test_is_retriable(self) -> None:
        """TemporaryResourceError should be a RetriableError."""
        assert issubclass(TemporaryResourceError, RetriableError)

    def test_instantiation(self) -> None:
        """TemporaryResourceError should be instantiable."""
        error = TemporaryResourceError("Memory pressure")
        assert "memory" in str(error).lower()

    def test_has_resource_type_attribute(self) -> None:
        """TemporaryResourceError should have resource_type attribute."""
        error = TemporaryResourceError("OOM", resource_type="memory")
        assert error.resource_type == "memory"


class TestQueueFullError:
    """Test QueueFullError for queue capacity issues."""

    def test_is_retriable(self) -> None:
        """QueueFullError should be a RetriableError."""
        assert issubclass(QueueFullError, RetriableError)

    def test_instantiation(self) -> None:
        """QueueFullError should be instantiable."""
        error = QueueFullError("Queue is full")
        assert "full" in str(error).lower()

    def test_has_max_concurrent_attribute(self) -> None:
        """QueueFullError should have max_concurrent attribute."""
        error = QueueFullError("Queue full", max_concurrent=10)
        assert error.max_concurrent == 10

    def test_has_current_size_attribute(self) -> None:
        """QueueFullError should have current_size attribute."""
        error = QueueFullError("Queue full", current_size=10, max_concurrent=10)
        assert error.current_size == 10


# =============================================================================
# Non-Retriable Exception Tests
# =============================================================================


class TestContextBudgetExceededError:
    """Test ContextBudgetExceededError for token limits."""

    def test_is_non_retriable(self) -> None:
        """ContextBudgetExceededError should be a NonRetriableError."""
        assert issubclass(ContextBudgetExceededError, NonRetriableError)

    def test_instantiation_with_tokens(self) -> None:
        """ContextBudgetExceededError should accept token counts."""
        error = ContextBudgetExceededError(current_tokens=18000, budget=16384)
        assert error.current_tokens == 18000
        assert error.budget == 16384

    def test_message_format(self) -> None:
        """ContextBudgetExceededError should have formatted message."""
        error = ContextBudgetExceededError(current_tokens=18000, budget=16384)
        assert "18000" in str(error)
        assert "16384" in str(error)

    def test_has_model_attribute(self) -> None:
        """ContextBudgetExceededError should support model attribute."""
        error = ContextBudgetExceededError(
            current_tokens=18000, budget=16384, model="phi-4"
        )
        assert error.model == "phi-4"


class TestCompressionFailedError:
    """Test CompressionFailedError for compression failures."""

    def test_is_non_retriable(self) -> None:
        """CompressionFailedError should be a NonRetriableError."""
        assert issubclass(CompressionFailedError, NonRetriableError)

    def test_instantiation(self) -> None:
        """CompressionFailedError should be instantiable."""
        error = CompressionFailedError("Could not compress to target")
        assert "compress" in str(error).lower()

    def test_has_iterations_attribute(self) -> None:
        """CompressionFailedError should support iterations attribute."""
        error = CompressionFailedError("Failed", iterations=5)
        assert error.iterations == 5

    def test_has_target_ratio_attribute(self) -> None:
        """CompressionFailedError should support target_ratio attribute."""
        error = CompressionFailedError("Failed", target_ratio=0.5)
        assert error.target_ratio == pytest.approx(0.5)


class TestHandoffStateInvalidError:
    """Test HandoffStateInvalidError for pipeline state issues."""

    def test_is_non_retriable(self) -> None:
        """HandoffStateInvalidError should be a NonRetriableError."""
        assert issubclass(HandoffStateInvalidError, NonRetriableError)

    def test_instantiation(self) -> None:
        """HandoffStateInvalidError should be instantiable."""
        error = HandoffStateInvalidError("Missing required field: goal")
        assert "goal" in str(error)

    def test_has_missing_fields_attribute(self) -> None:
        """HandoffStateInvalidError should support missing_fields attribute."""
        error = HandoffStateInvalidError("Invalid state", missing_fields=["goal", "step"])
        assert error.missing_fields == ["goal", "step"]


class TestModelNotFoundError:
    """Test ModelNotFoundError for unknown models."""

    def test_is_non_retriable(self) -> None:
        """ModelNotFoundError should be a NonRetriableError."""
        assert issubclass(ModelNotFoundError, NonRetriableError)

    def test_instantiation(self) -> None:
        """ModelNotFoundError should be instantiable."""
        error = ModelNotFoundError("Model not found: gpt-5")
        assert "gpt-5" in str(error)

    def test_has_model_id_attribute(self) -> None:
        """ModelNotFoundError should have model_id attribute."""
        error = ModelNotFoundError("Not found", model_id="gpt-5")
        assert error.model_id == "gpt-5"

    def test_has_available_models_attribute(self) -> None:
        """ModelNotFoundError should support available_models attribute."""
        error = ModelNotFoundError(
            "Not found",
            model_id="gpt-5",
            available_models=["phi-4", "llama-3.2-3b"]
        )
        assert error.available_models == ["phi-4", "llama-3.2-3b"]


class TestOrchestrationFailedError:
    """Test OrchestrationFailedError for orchestration issues."""

    def test_is_non_retriable(self) -> None:
        """OrchestrationFailedError should be a NonRetriableError."""
        assert issubclass(OrchestrationFailedError, NonRetriableError)

    def test_instantiation(self) -> None:
        """OrchestrationFailedError should be instantiable."""
        error = OrchestrationFailedError("No consensus reached")
        assert "consensus" in str(error).lower()

    def test_has_mode_attribute(self) -> None:
        """OrchestrationFailedError should support mode attribute."""
        error = OrchestrationFailedError("Failed", mode="ensemble")
        assert error.mode == "ensemble"

    def test_has_completed_steps_attribute(self) -> None:
        """OrchestrationFailedError should support completed_steps attribute."""
        error = OrchestrationFailedError("Failed", completed_steps=2, total_steps=3)
        assert error.completed_steps == 2
        assert error.total_steps == 3


class TestValidationError:
    """Test ValidationError for request validation issues."""

    def test_is_non_retriable(self) -> None:
        """ValidationError should be a NonRetriableError."""
        assert issubclass(ValidationError, NonRetriableError)

    def test_instantiation(self) -> None:
        """ValidationError should be instantiable."""
        error = ValidationError("Invalid temperature value")
        assert "temperature" in str(error).lower()

    def test_has_field_attribute(self) -> None:
        """ValidationError should have field attribute."""
        error = ValidationError("Invalid value", field="temperature")
        assert error.field == "temperature"

    def test_has_value_attribute(self) -> None:
        """ValidationError should have value attribute."""
        error = ValidationError("Out of range", field="temperature", value=2.5)
        assert error.value == pytest.approx(2.5)


class TestConfigurationError:
    """Test ConfigurationError for configuration issues."""

    def test_is_non_retriable(self) -> None:
        """ConfigurationError should be a NonRetriableError."""
        assert issubclass(ConfigurationError, NonRetriableError)

    def test_instantiation(self) -> None:
        """ConfigurationError should be instantiable."""
        error = ConfigurationError("Invalid configuration")
        assert "configuration" in str(error).lower()

    def test_has_setting_attribute(self) -> None:
        """ConfigurationError should support setting attribute."""
        error = ConfigurationError("Invalid setting", setting="INFERENCE_PORT")
        assert error.setting == "INFERENCE_PORT"


# =============================================================================
# AC-18.2: Exception Naming Convention Tests (AP-7)
# =============================================================================


class TestExceptionNamingConvention:
    """Test that all exceptions end in 'Error' (AP-7)."""

    @pytest.mark.parametrize("exc_class", [
        InferenceServiceError,
        RetriableError,
        NonRetriableError,
        ModelBusyError,
        ModelLoadingError,
        TemporaryResourceError,
        QueueFullError,
        ContextBudgetExceededError,
        CompressionFailedError,
        HandoffStateInvalidError,
        ModelNotFoundError,
        OrchestrationFailedError,
        ValidationError,
        ConfigurationError,
    ])
    def test_exception_name_ends_with_error(self, exc_class: type) -> None:
        """All exception class names should end with 'Error' (AP-7)."""
        assert exc_class.__name__.endswith("Error"), (
            f"Exception {exc_class.__name__} does not end with 'Error'"
        )

    @pytest.mark.parametrize("exc_class", [
        InferenceServiceError,
        RetriableError,
        NonRetriableError,
        ModelBusyError,
        ModelLoadingError,
        TemporaryResourceError,
        QueueFullError,
        ContextBudgetExceededError,
        CompressionFailedError,
        HandoffStateInvalidError,
        ModelNotFoundError,
        OrchestrationFailedError,
        ValidationError,
        ConfigurationError,
    ])
    def test_exception_name_matches_regex(self, exc_class: type) -> None:
        """All exception names should match .*Error$ regex (Exit Criteria)."""
        pattern = r".*Error$"
        assert re.match(pattern, exc_class.__name__), (
            f"Exception {exc_class.__name__} does not match {pattern}"
        )


# =============================================================================
# ErrorCode Enum Tests
# =============================================================================


class TestErrorCodeEnum:
    """Test ErrorCode enumeration."""

    def test_has_inference_error(self) -> None:
        """ErrorCode should have INFERENCE_ERROR value."""
        assert ErrorCode.INFERENCE_ERROR.value == "INFERENCE_ERROR"

    def test_has_model_not_found(self) -> None:
        """ErrorCode should have MODEL_NOT_FOUND value."""
        assert ErrorCode.MODEL_NOT_FOUND.value == "MODEL_NOT_FOUND"

    def test_has_context_budget_exceeded(self) -> None:
        """ErrorCode should have CONTEXT_BUDGET_EXCEEDED value."""
        assert ErrorCode.CONTEXT_BUDGET_EXCEEDED.value == "CONTEXT_BUDGET_EXCEEDED"

    def test_has_validation_error(self) -> None:
        """ErrorCode should have VALIDATION_ERROR value."""
        assert ErrorCode.VALIDATION_ERROR.value == "VALIDATION_ERROR"

    def test_has_queue_full(self) -> None:
        """ErrorCode should have QUEUE_FULL value."""
        assert ErrorCode.QUEUE_FULL.value == "QUEUE_FULL"

    def test_has_model_busy(self) -> None:
        """ErrorCode should have MODEL_BUSY value."""
        assert ErrorCode.MODEL_BUSY.value == "MODEL_BUSY"

    def test_has_orchestration_failed(self) -> None:
        """ErrorCode should have ORCHESTRATION_FAILED value."""
        assert ErrorCode.ORCHESTRATION_FAILED.value == "ORCHESTRATION_FAILED"

    def test_is_string_enum(self) -> None:
        """ErrorCode should be a string enum."""
        assert isinstance(ErrorCode.INFERENCE_ERROR, str)
        assert isinstance(ErrorCode.INFERENCE_ERROR.value, str)


# =============================================================================
# Exception Hierarchy Verification
# =============================================================================


class TestExceptionHierarchy:
    """Verify complete exception hierarchy structure."""

    def test_all_retriable_inherit_from_retriable_error(self) -> None:
        """All retriable exceptions should inherit from RetriableError."""
        retriable_classes = [
            ModelBusyError,
            ModelLoadingError,
            TemporaryResourceError,
            QueueFullError,
        ]
        for exc_class in retriable_classes:
            assert issubclass(exc_class, RetriableError), (
                f"{exc_class.__name__} should inherit from RetriableError"
            )

    def test_all_non_retriable_inherit_from_non_retriable_error(self) -> None:
        """All non-retriable exceptions should inherit from NonRetriableError."""
        non_retriable_classes = [
            ContextBudgetExceededError,
            CompressionFailedError,
            HandoffStateInvalidError,
            ModelNotFoundError,
            OrchestrationFailedError,
            ValidationError,
            ConfigurationError,
        ]
        for exc_class in non_retriable_classes:
            assert issubclass(exc_class, NonRetriableError), (
                f"{exc_class.__name__} should inherit from NonRetriableError"
            )

    def test_all_exceptions_inherit_from_base(self) -> None:
        """All custom exceptions should inherit from InferenceServiceError."""
        all_classes = [
            RetriableError,
            NonRetriableError,
            ModelBusyError,
            ModelLoadingError,
            TemporaryResourceError,
            QueueFullError,
            ContextBudgetExceededError,
            CompressionFailedError,
            HandoffStateInvalidError,
            ModelNotFoundError,
            OrchestrationFailedError,
            ValidationError,
            ConfigurationError,
        ]
        for exc_class in all_classes:
            assert issubclass(exc_class, InferenceServiceError), (
                f"{exc_class.__name__} should inherit from InferenceServiceError"
            )

    def test_can_catch_all_with_base_exception(self) -> None:
        """All exceptions should be catchable with InferenceServiceError."""
        exceptions_to_test = [
            ModelBusyError("busy"),
            ModelLoadingError("loading"),
            TemporaryResourceError("resource"),
            QueueFullError("full"),
            ContextBudgetExceededError(current_tokens=100, budget=50),
            CompressionFailedError("failed"),
            HandoffStateInvalidError("invalid"),
            ModelNotFoundError("not found"),
            OrchestrationFailedError("orchestration"),
            ValidationError("validation"),
            ConfigurationError("config"),
        ]

        for exc in exceptions_to_test:
            try:
                raise exc
            except InferenceServiceError as e:
                # Should be caught
                assert isinstance(e, InferenceServiceError)
            except Exception:
                pytest.fail(f"{type(exc).__name__} not caught by InferenceServiceError")

    def test_can_distinguish_retriable_from_non_retriable(self) -> None:
        """Should be able to distinguish retriable from non-retriable errors."""
        retriable = ModelBusyError("busy")
        non_retriable = ModelNotFoundError("not found")

        assert isinstance(retriable, RetriableError)
        assert not isinstance(retriable, NonRetriableError)

        assert isinstance(non_retriable, NonRetriableError)
        assert not isinstance(non_retriable, RetriableError)
