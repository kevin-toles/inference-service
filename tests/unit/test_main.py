"""Tests for main FastAPI application.

TDD RED Phase: These tests define the expected behavior of the FastAPI app.
Reference: WBS-INF2 Exit Criteria, CODING_PATTERNS_ANALYSIS.md

Tests verify:
- AC-2.3: FastAPI app initializes without errors
"""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


class TestAppInstance:
    """Test FastAPI app instance creation."""

    def test_app_is_fastapi_instance(self) -> None:
        """App is a FastAPI instance."""
        from src.main import app

        assert isinstance(app, FastAPI)

    def test_app_has_title(self) -> None:
        """App has a title configured."""
        from src.main import app

        assert app.title is not None
        assert len(app.title) > 0

    def test_app_has_version(self) -> None:
        """App has version configured."""
        from src.main import app

        assert app.version is not None


class TestAppLifespan:
    """Test FastAPI lifespan context manager."""

    def test_app_starts_without_error(self) -> None:
        """App startup runs without errors.

        AC-2.3: FastAPI app initializes without errors.
        """
        from src.main import app

        client = TestClient(app)
        # TestClient triggers lifespan startup/shutdown
        with client:
            pass  # Just verify no exception on startup

    def test_app_state_initialized_on_startup(self) -> None:
        """App state is set during lifespan startup."""
        from src.main import app

        client = TestClient(app)
        with client:
            assert hasattr(app.state, "initialized")
            assert app.state.initialized is True


class TestHealthEndpoint:
    """Test health check endpoint."""

    def test_health_endpoint_exists(self) -> None:
        """GET /health returns 200."""
        from src.main import app

        client = TestClient(app)
        with client:
            response = client.get("/health")
        assert response.status_code == 200

    def test_health_returns_status_ok(self) -> None:
        """Health endpoint returns status: ok."""
        from src.main import app

        client = TestClient(app)
        with client:
            response = client.get("/health")
        data = response.json()
        assert data["status"] == "ok"

    def test_health_returns_service_name(self) -> None:
        """Health endpoint returns service name."""
        from src.main import app

        client = TestClient(app)
        with client:
            response = client.get("/health")
        data = response.json()
        assert "service" in data
        assert data["service"] == "inference-service"


class TestReadyEndpoint:
    """Test readiness probe endpoint."""

    def test_ready_endpoint_exists(self) -> None:
        """GET /health/ready returns response (503 when no models loaded)."""
        from src.main import app

        client = TestClient(app)
        with client:
            response = client.get("/health/ready")
        # Returns 503 when no model manager is configured (WBS-INF7 AC-7.3)
        assert response.status_code == 503

    def test_ready_returns_not_ready_status(self) -> None:
        """Ready endpoint indicates not_ready state when no models."""
        from src.main import app

        client = TestClient(app)
        with client:
            response = client.get("/health/ready")
        data = response.json()
        # WBS-INF7 AC-7.3: Returns not_ready when no models loaded
        assert data["status"] == "not_ready"


class TestDocumentation:
    """Test OpenAPI documentation endpoints."""

    def test_docs_available_in_development(self) -> None:
        """Swagger docs available in development environment."""
        from src.main import app

        client = TestClient(app)
        with client:
            response = client.get("/docs")
        # Should be available (200) or redirect (307)
        assert response.status_code in (200, 307)

    def test_openapi_json_available(self) -> None:
        """OpenAPI JSON schema is available."""
        from src.main import app

        client = TestClient(app)
        with client:
            response = client.get("/openapi.json")
        assert response.status_code == 200
        data = response.json()
        assert "openapi" in data
        assert "paths" in data
