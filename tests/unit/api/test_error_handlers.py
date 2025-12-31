"""Unit tests for API error handlers.

Tests cover:
- AC-18.3: Error responses match llm-gateway schema
- AC-18.4: FastAPI exception handlers registered

Exit Criteria:
- Error response JSON matches ARCHITECTURE.md schema
- FastAPI returns proper status codes (400, 404, 500, 503)
"""

from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api.error_handlers import (
    ErrorResponse,
    ErrorDetail,
    register_exception_handlers,
    inference_service_error_handler,
    retriable_error_handler,
    non_retriable_error_handler,
    validation_error_handler,
    generic_error_handler,
)
from src.core.exceptions import (
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
    ErrorCode,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def app_with_handlers() -> FastAPI:
    """Create a FastAPI app with exception handlers registered."""
    app = FastAPI()
    register_exception_handlers(app)

    @app.get("/test/retriable")
    async def test_retriable() -> dict[str, str]:
        raise ModelBusyError("Model is busy", model_id="phi-4")

    @app.get("/test/non-retriable")
    async def test_non_retriable() -> dict[str, str]:
        raise ModelNotFoundError("Model not found", model_id="gpt-5")

    @app.get("/test/validation")
    async def test_validation() -> dict[str, str]:
        raise ValidationError("Invalid temperature", field="temperature", value=2.5)

    @app.get("/test/context-budget")
    async def test_context_budget() -> dict[str, str]:
        raise ContextBudgetExceededError(current_tokens=18000, budget=16384, model="phi-4")

    @app.get("/test/queue-full")
    async def test_queue_full() -> dict[str, str]:
        raise QueueFullError("Queue is full", max_concurrent=10, current_size=10)

    @app.get("/test/orchestration")
    async def test_orchestration() -> dict[str, str]:
        raise OrchestrationFailedError(
            "No consensus",
            mode="ensemble",
            completed_steps=2,
            total_steps=3,
        )

    @app.get("/test/generic")
    async def test_generic() -> dict[str, str]:
        raise RuntimeError("Unexpected error")

    return app


@pytest.fixture
def client(app_with_handlers: FastAPI) -> TestClient:
    """Create test client with exception handlers."""
    return TestClient(app_with_handlers, raise_server_exceptions=False)


# =============================================================================
# AC-18.3: Error Response Schema Tests
# =============================================================================


class TestErrorResponseSchema:
    """Test that error responses match llm-gateway schema."""

    def test_error_response_has_error_field(self, client: TestClient) -> None:
        """Error response should have top-level 'error' field."""
        response = client.get("/test/non-retriable")
        data = response.json()
        assert "error" in data

    def test_error_detail_has_code(self, client: TestClient) -> None:
        """Error detail should have 'code' field."""
        response = client.get("/test/non-retriable")
        data = response.json()
        assert "code" in data["error"]
        assert data["error"]["code"] == "MODEL_NOT_FOUND"

    def test_error_detail_has_message(self, client: TestClient) -> None:
        """Error detail should have 'message' field."""
        response = client.get("/test/non-retriable")
        data = response.json()
        assert "message" in data["error"]
        assert "not found" in data["error"]["message"].lower()

    def test_error_detail_has_type(self, client: TestClient) -> None:
        """Error detail should have 'type' field (retriable/non_retriable)."""
        response = client.get("/test/non-retriable")
        data = response.json()
        assert "type" in data["error"]
        assert data["error"]["type"] == "non_retriable"

    def test_error_detail_has_provider(self, client: TestClient) -> None:
        """Error detail should have 'provider' field (inference-service)."""
        response = client.get("/test/non-retriable")
        data = response.json()
        assert "provider" in data["error"]
        assert data["error"]["provider"] == "inference-service"

    def test_error_detail_has_details(self, client: TestClient) -> None:
        """Error detail should have 'details' field with context."""
        response = client.get("/test/context-budget")
        data = response.json()
        assert "details" in data["error"]
        details = data["error"]["details"]
        assert details["current_tokens"] == 18000
        assert details["budget"] == 16384
        assert details["model"] == "phi-4"


class TestErrorResponseModels:
    """Test Pydantic models for error responses."""

    def test_error_detail_model(self) -> None:
        """ErrorDetail model should be valid."""
        detail = ErrorDetail(
            code="MODEL_NOT_FOUND",
            message="Model not found",
            type="non_retriable",
            provider="inference-service",
            details={"model_id": "gpt-5"},
        )
        assert detail.code == "MODEL_NOT_FOUND"
        assert detail.type == "non_retriable"

    def test_error_response_model(self) -> None:
        """ErrorResponse model should wrap ErrorDetail."""
        detail = ErrorDetail(
            code="MODEL_NOT_FOUND",
            message="Model not found",
            type="non_retriable",
            provider="inference-service",
        )
        response = ErrorResponse(error=detail)
        assert response.error.code == "MODEL_NOT_FOUND"

    def test_error_response_json_matches_schema(self) -> None:
        """ErrorResponse JSON should match ARCHITECTURE.md schema."""
        detail = ErrorDetail(
            code="CONTEXT_BUDGET_EXCEEDED",
            message="Content cannot fit in phi-4 context window",
            type="non_retriable",
            provider="inference-service",
            details={
                "current_tokens": 18000,
                "budget": 16384,
                "model": "phi-4",
                "compression_attempted": True,
            },
        )
        response = ErrorResponse(error=detail)
        json_data = response.model_dump()

        # Verify structure matches ARCHITECTURE.md
        assert "error" in json_data
        assert json_data["error"]["code"] == "CONTEXT_BUDGET_EXCEEDED"
        assert json_data["error"]["type"] == "non_retriable"
        assert json_data["error"]["provider"] == "inference-service"
        assert json_data["error"]["details"]["current_tokens"] == 18000


# =============================================================================
# AC-18.4: Status Code Tests
# =============================================================================


class TestStatusCodes:
    """Test FastAPI returns proper HTTP status codes."""

    def test_model_not_found_returns_404(self, client: TestClient) -> None:
        """ModelNotFoundError should return 404."""
        response = client.get("/test/non-retriable")
        assert response.status_code == 404

    def test_validation_error_returns_400(self, client: TestClient) -> None:
        """ValidationError should return 400."""
        response = client.get("/test/validation")
        assert response.status_code == 400

    def test_context_budget_exceeded_returns_400(self, client: TestClient) -> None:
        """ContextBudgetExceededError should return 400."""
        response = client.get("/test/context-budget")
        assert response.status_code == 400

    def test_queue_full_returns_503(self, client: TestClient) -> None:
        """QueueFullError should return 503 (Service Unavailable)."""
        response = client.get("/test/queue-full")
        assert response.status_code == 503

    def test_model_busy_returns_503(self, client: TestClient) -> None:
        """ModelBusyError should return 503 (Service Unavailable)."""
        response = client.get("/test/retriable")
        assert response.status_code == 503

    def test_orchestration_failed_returns_500(self, client: TestClient) -> None:
        """OrchestrationFailedError should return 500."""
        response = client.get("/test/orchestration")
        assert response.status_code == 500

    def test_generic_error_returns_500(self, client: TestClient) -> None:
        """Generic exceptions should return 500."""
        response = client.get("/test/generic")
        assert response.status_code == 500


class TestRetriableErrorResponses:
    """Test retriable error responses include retry information."""

    def test_retry_after_header(self, client: TestClient) -> None:
        """Retriable errors should include Retry-After header."""
        response = client.get("/test/retriable")
        assert "Retry-After" in response.headers

    def test_retry_after_value(self, client: TestClient) -> None:
        """Retry-After header should have correct value."""
        response = client.get("/test/retriable")
        # Default retry_after_ms is 1000, so 1 second
        assert response.headers.get("Retry-After") is not None

    def test_retriable_type_in_response(self, client: TestClient) -> None:
        """Retriable error type should be 'retriable'."""
        response = client.get("/test/retriable")
        data = response.json()
        assert data["error"]["type"] == "retriable"

    def test_queue_full_includes_retry_after(self, client: TestClient) -> None:
        """QueueFullError should include Retry-After header."""
        response = client.get("/test/queue-full")
        assert "Retry-After" in response.headers


# =============================================================================
# Error Handler Function Tests
# =============================================================================


class TestErrorHandlerFunctions:
    """Test individual error handler functions."""

    @pytest.fixture
    def mock_request(self) -> Any:
        """Create a mock request for testing handlers."""
        from unittest.mock import MagicMock
        return MagicMock(spec=["url", "method"])

    def test_inference_service_error_handler(self, mock_request: Any) -> None:
        """Test base InferenceServiceError handler."""
        error = InferenceServiceError("Test error")
        response = inference_service_error_handler(mock_request, error)
        assert response.status_code == 500

    def test_retriable_error_handler(self, mock_request: Any) -> None:
        """Test RetriableError handler."""
        error = ModelBusyError("Busy", model_id="phi-4", retry_after_ms=2000)
        response = retriable_error_handler(mock_request, error)
        assert response.status_code == 503
        assert "Retry-After" in response.headers

    def test_non_retriable_error_handler(self, mock_request: Any) -> None:
        """Test NonRetriableError handler."""
        error = ModelNotFoundError("Not found", model_id="gpt-5")
        response = non_retriable_error_handler(mock_request, error)
        assert response.status_code == 404

    def test_validation_error_handler(self, mock_request: Any) -> None:
        """Test ValidationError handler."""
        error = ValidationError("Invalid", field="temperature")
        response = validation_error_handler(mock_request, error)
        assert response.status_code == 400

    def test_generic_error_handler(self, mock_request: Any) -> None:
        """Test generic Exception handler."""
        error = RuntimeError("Unexpected")
        response = generic_error_handler(mock_request, error)
        assert response.status_code == 500


# =============================================================================
# Error Details Extraction Tests
# =============================================================================


class TestErrorDetailsExtraction:
    """Test that error details are properly extracted."""

    def test_context_budget_details(self, client: TestClient) -> None:
        """ContextBudgetExceededError details should be extracted."""
        response = client.get("/test/context-budget")
        details = response.json()["error"]["details"]
        assert details["current_tokens"] == 18000
        assert details["budget"] == 16384
        assert details["model"] == "phi-4"

    def test_queue_full_details(self, client: TestClient) -> None:
        """QueueFullError details should be extracted."""
        response = client.get("/test/queue-full")
        details = response.json()["error"]["details"]
        assert details["max_concurrent"] == 10
        assert details["current_size"] == 10

    def test_orchestration_failed_details(self, client: TestClient) -> None:
        """OrchestrationFailedError details should be extracted."""
        response = client.get("/test/orchestration")
        details = response.json()["error"]["details"]
        assert details["mode"] == "ensemble"
        assert details["completed_steps"] == 2
        assert details["total_steps"] == 3

    def test_validation_error_details(self, client: TestClient) -> None:
        """ValidationError details should be extracted."""
        response = client.get("/test/validation")
        details = response.json()["error"]["details"]
        assert details["field"] == "temperature"
        assert details["value"] == 2.5


# =============================================================================
# Handler Registration Tests
# =============================================================================


class TestHandlerRegistration:
    """Test exception handler registration."""

    def test_register_exception_handlers(self) -> None:
        """register_exception_handlers should add handlers to app."""
        app = FastAPI()
        assert len(app.exception_handlers) == 0 or Exception not in app.exception_handlers

        register_exception_handlers(app)

        # Should have handlers for our custom exceptions
        assert InferenceServiceError in app.exception_handlers
        assert RetriableError in app.exception_handlers
        assert NonRetriableError in app.exception_handlers
        assert Exception in app.exception_handlers

    def test_more_specific_handlers_take_precedence(
        self, app_with_handlers: FastAPI
    ) -> None:
        """More specific exception handlers should take precedence."""
        client = TestClient(app_with_handlers)

        # ModelNotFoundError is NonRetriable, should get 404 not 500
        response = client.get("/test/non-retriable")
        assert response.status_code == 404

        # ValidationError should get 400
        response = client.get("/test/validation")
        assert response.status_code == 400


# =============================================================================
# Integration with Main App Tests
# =============================================================================


class TestMainAppIntegration:
    """Test that error handlers work with the main app."""

    def test_import_from_error_handlers(self) -> None:
        """Should be able to import all required components."""
        from src.api.error_handlers import (
            ErrorResponse,
            ErrorDetail,
            register_exception_handlers,
        )

        assert ErrorResponse is not None
        assert ErrorDetail is not None
        assert callable(register_exception_handlers)

    def test_error_response_is_json_serializable(self) -> None:
        """ErrorResponse should be JSON serializable."""
        detail = ErrorDetail(
            code="TEST_ERROR",
            message="Test message",
            type="non_retriable",
            provider="inference-service",
        )
        response = ErrorResponse(error=detail)

        # Should not raise
        json_str = response.model_dump_json()
        assert "TEST_ERROR" in json_str
