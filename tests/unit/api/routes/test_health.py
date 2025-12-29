"""Unit tests for health API routes.

Tests the /health (liveness) and /health/ready (readiness) endpoints.

Reference: WBS-INF7 AC-7.1 through AC-7.4
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI, status
from fastapi.testclient import TestClient


# =============================================================================
# Constants (S1192: Avoid duplicated string literals)
# =============================================================================

STATUS_OK = "ok"
STATUS_READY = "ready"
STATUS_NOT_READY = "not_ready"
HEALTH_ENDPOINT = "/health"
READY_ENDPOINT = "/health/ready"
MODEL_PHI4 = "phi-4"
MODEL_DEEPSEEK = "deepseek-r1-7b"
CONFIG_D3 = "D3"


# =============================================================================
# TestHealthRouterImport
# =============================================================================


class TestHealthRouterImport:
    """Test that health router can be imported."""

    def test_health_router_importable(self) -> None:
        """AC-7.1: Health router exists and can be imported."""
        from src.api.routes.health import router

        assert router is not None

    def test_health_response_models_importable(self) -> None:
        """AC-7.1: Response models can be imported."""
        from src.api.routes.health import (
            HealthResponse,
            ReadinessResponse,
        )

        assert HealthResponse is not None
        assert ReadinessResponse is not None


# =============================================================================
# TestHealthEndpoint
# =============================================================================


class TestHealthEndpoint:
    """Test /health liveness endpoint (AC-7.1)."""

    @pytest.fixture
    def app(self) -> FastAPI:
        """Create FastAPI app with health router."""
        from src.api.routes.health import router

        app = FastAPI()
        app.include_router(router)
        return app

    @pytest.fixture
    def client(self, app: FastAPI) -> TestClient:
        """Create test client."""
        return TestClient(app)

    def test_health_returns_200(self, client: TestClient) -> None:
        """AC-7.1: GET /health returns 200 when service is up."""
        response = client.get(HEALTH_ENDPOINT)

        assert response.status_code == status.HTTP_200_OK

    def test_health_returns_status_ok(self, client: TestClient) -> None:
        """AC-7.1: GET /health returns status 'ok'."""
        response = client.get(HEALTH_ENDPOINT)
        data = response.json()

        assert data["status"] == STATUS_OK

    def test_health_includes_service_name(self, client: TestClient) -> None:
        """AC-7.1: Health response includes service name."""
        response = client.get(HEALTH_ENDPOINT)
        data = response.json()

        assert "service" in data
        assert data["service"] == "inference-service"

    def test_health_includes_version(self, client: TestClient) -> None:
        """AC-7.1: Health response includes version."""
        response = client.get(HEALTH_ENDPOINT)
        data = response.json()

        assert "version" in data

    def test_health_response_schema(self, client: TestClient) -> None:
        """AC-7.1: Health response matches expected schema."""
        response = client.get(HEALTH_ENDPOINT)
        data = response.json()

        # Required fields
        assert "status" in data
        assert "service" in data
        assert "version" in data


# =============================================================================
# TestReadinessEndpointReady
# =============================================================================


class TestReadinessEndpointReady:
    """Test /health/ready when models are loaded (AC-7.2, AC-7.4)."""

    @pytest.fixture
    def mock_model_manager(self) -> MagicMock:
        """Create mock ModelManager with loaded models."""
        manager = MagicMock()
        manager.get_loaded_models.return_value = [MODEL_PHI4, MODEL_DEEPSEEK]
        manager.get_available_models.return_value = [MODEL_PHI4, MODEL_DEEPSEEK]
        return manager

    @pytest.fixture
    def app_with_models(self, mock_model_manager: MagicMock) -> FastAPI:
        """Create FastAPI app with mock model manager."""
        from src.api.routes.health import router

        app = FastAPI()
        app.include_router(router)
        app.state.model_manager = mock_model_manager
        app.state.config_preset = CONFIG_D3
        app.state.orchestration_mode = "debate"
        return app

    @pytest.fixture
    def client_ready(self, app_with_models: FastAPI) -> TestClient:
        """Create test client with models loaded."""
        return TestClient(app_with_models)

    def test_ready_returns_200_when_models_loaded(
        self, client_ready: TestClient
    ) -> None:
        """AC-7.2: GET /health/ready returns 200 when models loaded."""
        response = client_ready.get(READY_ENDPOINT)

        assert response.status_code == status.HTTP_200_OK

    def test_ready_returns_status_ready(self, client_ready: TestClient) -> None:
        """AC-7.2: Status is 'ready' when models loaded."""
        response = client_ready.get(READY_ENDPOINT)
        data = response.json()

        assert data["status"] == STATUS_READY

    def test_ready_includes_loaded_models(self, client_ready: TestClient) -> None:
        """AC-7.4: Health response includes loaded_models list."""
        response = client_ready.get(READY_ENDPOINT)
        data = response.json()

        assert "loaded_models" in data
        assert MODEL_PHI4 in data["loaded_models"]
        assert MODEL_DEEPSEEK in data["loaded_models"]

    def test_ready_includes_config_preset(self, client_ready: TestClient) -> None:
        """AC-7.4: Ready response includes config preset."""
        response = client_ready.get(READY_ENDPOINT)
        data = response.json()

        assert "config" in data
        assert data["config"] == CONFIG_D3

    def test_ready_includes_orchestration_mode(
        self, client_ready: TestClient
    ) -> None:
        """AC-7.4: Ready response includes orchestration mode."""
        response = client_ready.get(READY_ENDPOINT)
        data = response.json()

        assert "orchestration_mode" in data
        assert data["orchestration_mode"] == "debate"

    def test_ready_response_schema_when_ready(
        self, client_ready: TestClient
    ) -> None:
        """AC-7.2, AC-7.4: Ready response matches expected schema."""
        response = client_ready.get(READY_ENDPOINT)
        data = response.json()

        assert data["status"] == STATUS_READY
        assert "config" in data
        assert "loaded_models" in data
        assert "orchestration_mode" in data


# =============================================================================
# TestReadinessEndpointNotReady
# =============================================================================


class TestReadinessEndpointNotReady:
    """Test /health/ready when no models loaded (AC-7.3)."""

    @pytest.fixture
    def mock_model_manager_empty(self) -> MagicMock:
        """Create mock ModelManager with no loaded models."""
        manager = MagicMock()
        manager.get_loaded_models.return_value = []
        manager.get_available_models.return_value = [MODEL_PHI4, MODEL_DEEPSEEK]
        return manager

    @pytest.fixture
    def app_no_models(self, mock_model_manager_empty: MagicMock) -> FastAPI:
        """Create FastAPI app with no models loaded."""
        from src.api.routes.health import router

        app = FastAPI()
        app.include_router(router)
        app.state.model_manager = mock_model_manager_empty
        app.state.config_preset = None
        app.state.orchestration_mode = None
        return app

    @pytest.fixture
    def client_not_ready(self, app_no_models: FastAPI) -> TestClient:
        """Create test client with no models loaded."""
        return TestClient(app_no_models)

    def test_ready_returns_503_when_no_models(
        self, client_not_ready: TestClient
    ) -> None:
        """AC-7.3: GET /health/ready returns 503 when no models loaded."""
        response = client_not_ready.get(READY_ENDPOINT)

        assert response.status_code == status.HTTP_503_SERVICE_UNAVAILABLE

    def test_ready_returns_status_not_ready(
        self, client_not_ready: TestClient
    ) -> None:
        """AC-7.3: Status is 'not_ready' when no models loaded."""
        response = client_not_ready.get(READY_ENDPOINT)
        data = response.json()

        assert data["status"] == STATUS_NOT_READY

    def test_ready_includes_reason_when_not_ready(
        self, client_not_ready: TestClient
    ) -> None:
        """AC-7.3: Response includes reason when not ready."""
        response = client_not_ready.get(READY_ENDPOINT)
        data = response.json()

        assert "reason" in data

    def test_ready_includes_empty_models_list(
        self, client_not_ready: TestClient
    ) -> None:
        """AC-7.3, AC-7.4: Response includes empty loaded_models list."""
        response = client_not_ready.get(READY_ENDPOINT)
        data = response.json()

        assert "loaded_models" in data
        assert data["loaded_models"] == []


# =============================================================================
# TestReadinessEndpointNoManager
# =============================================================================


class TestReadinessEndpointNoManager:
    """Test /health/ready when model manager not initialized."""

    @pytest.fixture
    def app_no_manager(self) -> FastAPI:
        """Create FastAPI app without model manager."""
        from src.api.routes.health import router

        app = FastAPI()
        app.include_router(router)
        # No model_manager in app.state
        return app

    @pytest.fixture
    def client_no_manager(self, app_no_manager: FastAPI) -> TestClient:
        """Create test client without model manager."""
        return TestClient(app_no_manager)

    def test_ready_returns_503_when_no_manager(
        self, client_no_manager: TestClient
    ) -> None:
        """AC-7.3: Returns 503 when model manager not initialized."""
        response = client_no_manager.get(READY_ENDPOINT)

        assert response.status_code == status.HTTP_503_SERVICE_UNAVAILABLE

    def test_ready_indicates_not_initialized(
        self, client_no_manager: TestClient
    ) -> None:
        """AC-7.3: Response indicates service not initialized."""
        response = client_no_manager.get(READY_ENDPOINT)
        data = response.json()

        assert data["status"] == STATUS_NOT_READY
        assert "reason" in data


# =============================================================================
# TestReadinessEndpointLoading
# =============================================================================


class TestReadinessEndpointLoading:
    """Test /health/ready during model loading (partial load)."""

    @pytest.fixture
    def mock_model_manager_partial(self) -> MagicMock:
        """Create mock ModelManager with partial load."""
        manager = MagicMock()
        manager.get_loaded_models.return_value = [MODEL_PHI4]  # 1 of 2 loaded
        manager.get_available_models.return_value = [MODEL_PHI4, MODEL_DEEPSEEK]
        return manager

    @pytest.fixture
    def app_partial_load(self, mock_model_manager_partial: MagicMock) -> FastAPI:
        """Create FastAPI app with partial model load."""
        from src.api.routes.health import router

        app = FastAPI()
        app.include_router(router)
        app.state.model_manager = mock_model_manager_partial
        app.state.config_preset = CONFIG_D3
        app.state.orchestration_mode = "debate"
        app.state.expected_models = [MODEL_PHI4, MODEL_DEEPSEEK]
        return app

    @pytest.fixture
    def client_partial(self, app_partial_load: FastAPI) -> TestClient:
        """Create test client with partial load."""
        return TestClient(app_partial_load)

    def test_ready_returns_200_with_partial_load(
        self, client_partial: TestClient
    ) -> None:
        """AC-7.2: Returns 200 if at least one model is loaded."""
        response = client_partial.get(READY_ENDPOINT)

        # Still ready if at least one model loaded
        assert response.status_code == status.HTTP_200_OK

    def test_ready_shows_partial_models(self, client_partial: TestClient) -> None:
        """AC-7.4: Shows only the loaded models."""
        response = client_partial.get(READY_ENDPOINT)
        data = response.json()

        assert MODEL_PHI4 in data["loaded_models"]
        assert len(data["loaded_models"]) == 1


# =============================================================================
# TestHealthResponseModels
# =============================================================================


class TestHealthResponseModels:
    """Test Pydantic response models."""

    def test_health_response_model_fields(self) -> None:
        """Health response model has required fields."""
        from src.api.routes.health import HealthResponse

        response = HealthResponse(
            status=STATUS_OK,
            service="inference-service",
            version="0.1.0",
        )

        assert response.status == STATUS_OK
        assert response.service == "inference-service"
        assert response.version == "0.1.0"

    def test_readiness_response_model_ready(self) -> None:
        """Readiness response model for ready state."""
        from src.api.routes.health import ReadinessResponse

        response = ReadinessResponse(
            status=STATUS_READY,
            config=CONFIG_D3,
            loaded_models=[MODEL_PHI4, MODEL_DEEPSEEK],
            orchestration_mode="debate",
        )

        assert response.status == STATUS_READY
        assert response.config == CONFIG_D3
        assert MODEL_PHI4 in response.loaded_models

    def test_readiness_response_model_not_ready(self) -> None:
        """Readiness response model for not ready state."""
        from src.api.routes.health import ReadinessResponse

        response = ReadinessResponse(
            status=STATUS_NOT_READY,
            loaded_models=[],
            reason="No models loaded",
        )

        assert response.status == STATUS_NOT_READY
        assert response.reason == "No models loaded"
        assert response.loaded_models == []


# =============================================================================
# TestKubernetesProbes
# =============================================================================


class TestKubernetesProbes:
    """Test endpoints work for Kubernetes probes."""

    @pytest.fixture
    def mock_model_manager(self) -> MagicMock:
        """Create mock ModelManager."""
        manager = MagicMock()
        manager.get_loaded_models.return_value = [MODEL_PHI4]
        return manager

    @pytest.fixture
    def app(self, mock_model_manager: MagicMock) -> FastAPI:
        """Create FastAPI app."""
        from src.api.routes.health import router

        app = FastAPI()
        app.include_router(router)
        app.state.model_manager = mock_model_manager
        app.state.config_preset = "S1"
        app.state.orchestration_mode = "single"
        return app

    @pytest.fixture
    def client(self, app: FastAPI) -> TestClient:
        """Create test client."""
        return TestClient(app)

    def test_liveness_probe_path(self, client: TestClient) -> None:
        """Liveness probe uses /health path."""
        response = client.get("/health")
        assert response.status_code == status.HTTP_200_OK

    def test_readiness_probe_path(self, client: TestClient) -> None:
        """Readiness probe uses /health/ready path."""
        response = client.get("/health/ready")
        assert response.status_code == status.HTTP_200_OK

    def test_health_response_is_json(self, client: TestClient) -> None:
        """Health endpoints return JSON."""
        response = client.get("/health")
        assert response.headers["content-type"] == "application/json"

    def test_ready_response_is_json(self, client: TestClient) -> None:
        """Ready endpoints return JSON."""
        response = client.get("/health/ready")
        assert response.headers["content-type"] == "application/json"
