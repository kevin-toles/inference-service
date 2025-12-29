"""Unit tests for models API routes.

Tests the /v1/models endpoint for listing, loading, and unloading models.

Reference: WBS-INF8 AC-8.1 through AC-8.4
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI, status
from fastapi.testclient import TestClient


# =============================================================================
# Constants (S1192: Avoid duplicated string literals)
# =============================================================================

MODEL_PHI4 = "phi-4"
MODEL_DEEPSEEK = "deepseek-r1-7b"
MODEL_QWEN = "qwen2.5-7b"
MODEL_LLAMA = "llama-3.2-3b"
MODELS_ENDPOINT = "/v1/models"
CONFIG_D3 = "D3"
STATUS_LOADED = "loaded"
STATUS_AVAILABLE = "available"


# =============================================================================
# TestModelsRouterImport
# =============================================================================


class TestModelsRouterImport:
    """Test that models router can be imported."""

    def test_models_router_importable(self) -> None:
        """AC-8.1: Models router exists and can be imported."""
        from src.api.routes.models import router

        assert router is not None

    def test_models_response_models_importable(self) -> None:
        """AC-8.1: Response models can be imported."""
        from src.api.routes.models import (
            ModelInfo,
            ModelsListResponse,
        )

        assert ModelInfo is not None
        assert ModelsListResponse is not None


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def mock_model_manager() -> MagicMock:
    """Create mock ModelManager with models."""
    manager = MagicMock()

    # Set up model info returns
    phi4_info = MagicMock()
    phi4_info.model_id = MODEL_PHI4
    phi4_info.name = "Microsoft Phi-4"
    phi4_info.size_gb = 8.4
    phi4_info.context_length = 16384
    phi4_info.roles = ["primary", "thinker", "coder"]
    phi4_info.status = STATUS_LOADED

    deepseek_info = MagicMock()
    deepseek_info.model_id = MODEL_DEEPSEEK
    deepseek_info.name = "DeepSeek R1 Distill 7B"
    deepseek_info.size_gb = 4.7
    deepseek_info.context_length = 32768
    deepseek_info.roles = ["thinker"]
    deepseek_info.status = STATUS_AVAILABLE

    manager.list_all_models.return_value = [phi4_info, deepseek_info]
    manager.get_loaded_models.return_value = [MODEL_PHI4]
    manager.get_available_models.return_value = [MODEL_PHI4, MODEL_DEEPSEEK]
    manager.get_model_info.side_effect = lambda model_id: {
        MODEL_PHI4: phi4_info,
        MODEL_DEEPSEEK: deepseek_info,
    }.get(model_id)

    # Async methods need AsyncMock
    manager.load_model = AsyncMock()
    manager.unload_model = AsyncMock()

    return manager


@pytest.fixture
def app_with_manager(mock_model_manager: MagicMock) -> FastAPI:
    """Create FastAPI app with mock model manager."""
    from src.api.routes.models import router

    app = FastAPI()
    app.include_router(router, prefix="/v1")
    app.state.model_manager = mock_model_manager
    app.state.config_preset = CONFIG_D3
    app.state.orchestration_mode = "debate"
    return app


@pytest.fixture
def client(app_with_manager: FastAPI) -> TestClient:
    """Create test client."""
    return TestClient(app_with_manager)


# =============================================================================
# TestListModels
# =============================================================================


class TestListModels:
    """Test GET /v1/models endpoint (AC-8.1, AC-8.4)."""

    def test_list_models_returns_200(self, client: TestClient) -> None:
        """AC-8.1: GET /v1/models returns 200."""
        response = client.get(MODELS_ENDPOINT)

        assert response.status_code == status.HTTP_200_OK

    def test_list_models_returns_data_array(self, client: TestClient) -> None:
        """AC-8.1: Response contains data array."""
        response = client.get(MODELS_ENDPOINT)
        data = response.json()

        assert "data" in data
        assert isinstance(data["data"], list)

    def test_list_models_includes_all_models(self, client: TestClient) -> None:
        """AC-8.1: GET /v1/models lists available and loaded models."""
        response = client.get(MODELS_ENDPOINT)
        data = response.json()

        model_ids = [m["id"] for m in data["data"]]
        assert MODEL_PHI4 in model_ids
        assert MODEL_DEEPSEEK in model_ids

    def test_list_models_includes_status(self, client: TestClient) -> None:
        """AC-8.4: Model response includes status."""
        response = client.get(MODELS_ENDPOINT)
        data = response.json()

        for model in data["data"]:
            assert "status" in model
            assert model["status"] in [STATUS_LOADED, STATUS_AVAILABLE]

    def test_list_models_includes_memory(self, client: TestClient) -> None:
        """AC-8.4: Model response includes memory."""
        response = client.get(MODELS_ENDPOINT)
        data = response.json()

        for model in data["data"]:
            assert "memory_mb" in model
            assert isinstance(model["memory_mb"], int)

    def test_list_models_includes_context(self, client: TestClient) -> None:
        """AC-8.4: Model response includes context length."""
        response = client.get(MODELS_ENDPOINT)
        data = response.json()

        for model in data["data"]:
            assert "context_length" in model
            assert isinstance(model["context_length"], int)

    def test_list_models_includes_roles(self, client: TestClient) -> None:
        """AC-8.4: Model response includes roles."""
        response = client.get(MODELS_ENDPOINT)
        data = response.json()

        for model in data["data"]:
            assert "roles" in model
            assert isinstance(model["roles"], list)

    def test_list_models_includes_config(self, client: TestClient) -> None:
        """AC-8.4: Response includes config preset."""
        response = client.get(MODELS_ENDPOINT)
        data = response.json()

        assert "config" in data
        assert data["config"] == CONFIG_D3

    def test_list_models_includes_orchestration_mode(
        self, client: TestClient
    ) -> None:
        """AC-8.4: Response includes orchestration mode."""
        response = client.get(MODELS_ENDPOINT)
        data = response.json()

        assert "orchestration_mode" in data
        assert data["orchestration_mode"] == "debate"

    def test_list_models_schema_matches_architecture(
        self, client: TestClient
    ) -> None:
        """AC-8.4: Response matches ARCHITECTURE.md schema."""
        response = client.get(MODELS_ENDPOINT)
        data = response.json()

        # Verify phi-4 model structure
        phi4 = next((m for m in data["data"] if m["id"] == MODEL_PHI4), None)
        assert phi4 is not None
        assert phi4["status"] == STATUS_LOADED
        assert phi4["memory_mb"] == 8400
        assert phi4["context_length"] == 16384
        assert "primary" in phi4["roles"]


# =============================================================================
# TestListModelsNoManager
# =============================================================================


class TestListModelsNoManager:
    """Test GET /v1/models when model manager not initialized."""

    @pytest.fixture
    def app_no_manager(self) -> FastAPI:
        """Create FastAPI app without model manager."""
        from src.api.routes.models import router

        app = FastAPI()
        app.include_router(router, prefix="/v1")
        return app

    @pytest.fixture
    def client_no_manager(self, app_no_manager: FastAPI) -> TestClient:
        """Create test client without model manager."""
        return TestClient(app_no_manager)

    def test_list_models_returns_503_when_no_manager(
        self, client_no_manager: TestClient
    ) -> None:
        """Returns 503 when model manager not initialized."""
        response = client_no_manager.get(MODELS_ENDPOINT)

        assert response.status_code == status.HTTP_503_SERVICE_UNAVAILABLE


# =============================================================================
# TestLoadModel
# =============================================================================


class TestLoadModel:
    """Test POST /v1/models/{id}/load endpoint (AC-8.2)."""

    def test_load_model_returns_200(
        self, client: TestClient, mock_model_manager: MagicMock
    ) -> None:
        """AC-8.2: POST /v1/models/{id}/load returns 200 on success."""
        response = client.post(f"{MODELS_ENDPOINT}/{MODEL_DEEPSEEK}/load")

        assert response.status_code == status.HTTP_200_OK

    def test_load_model_calls_manager(
        self, client: TestClient, mock_model_manager: MagicMock
    ) -> None:
        """AC-8.2: Load endpoint calls model manager."""
        client.post(f"{MODELS_ENDPOINT}/{MODEL_DEEPSEEK}/load")

        mock_model_manager.load_model.assert_called_once_with(MODEL_DEEPSEEK)

    def test_load_model_returns_model_info(
        self, client: TestClient, mock_model_manager: MagicMock
    ) -> None:
        """AC-8.2: Load endpoint returns updated model info."""
        response = client.post(f"{MODELS_ENDPOINT}/{MODEL_DEEPSEEK}/load")
        data = response.json()

        assert "id" in data
        assert "status" in data

    def test_load_model_invalid_id_returns_404(
        self, client: TestClient, mock_model_manager: MagicMock
    ) -> None:
        """AC-8.2: Invalid model_id returns 404."""
        from src.services.model_manager import ModelNotAvailableError

        mock_model_manager.load_model.side_effect = ModelNotAvailableError(
            "Model 'invalid-model' not found"
        )

        response = client.post(f"{MODELS_ENDPOINT}/invalid-model/load")

        assert response.status_code == status.HTTP_404_NOT_FOUND

    def test_load_model_404_has_error_message(
        self, client: TestClient, mock_model_manager: MagicMock
    ) -> None:
        """AC-8.2: 404 response includes clear error message."""
        from src.services.model_manager import ModelNotAvailableError

        mock_model_manager.load_model.side_effect = ModelNotAvailableError(
            "Model 'invalid-model' not found"
        )

        response = client.post(f"{MODELS_ENDPOINT}/invalid-model/load")
        data = response.json()

        assert "detail" in data
        assert "invalid-model" in data["detail"]

    def test_load_model_memory_exceeded_returns_507(
        self, client: TestClient, mock_model_manager: MagicMock
    ) -> None:
        """AC-8.2: Memory limit exceeded returns 507."""
        from src.services.model_manager import MemoryLimitExceededError

        mock_model_manager.load_model.side_effect = MemoryLimitExceededError(
            "Loading would exceed memory limit"
        )

        response = client.post(f"{MODELS_ENDPOINT}/{MODEL_DEEPSEEK}/load")

        assert response.status_code == status.HTTP_507_INSUFFICIENT_STORAGE


# =============================================================================
# TestLoadModelNoManager
# =============================================================================


class TestLoadModelNoManager:
    """Test POST /v1/models/{id}/load when no model manager."""

    @pytest.fixture
    def app_no_manager(self) -> FastAPI:
        """Create FastAPI app without model manager."""
        from src.api.routes.models import router

        app = FastAPI()
        app.include_router(router, prefix="/v1")
        return app

    @pytest.fixture
    def client_no_manager(self, app_no_manager: FastAPI) -> TestClient:
        """Create test client without model manager."""
        return TestClient(app_no_manager)

    def test_load_returns_503_when_no_manager(
        self, client_no_manager: TestClient
    ) -> None:
        """Returns 503 when model manager not initialized."""
        response = client_no_manager.post(f"{MODELS_ENDPOINT}/{MODEL_PHI4}/load")

        assert response.status_code == status.HTTP_503_SERVICE_UNAVAILABLE


# =============================================================================
# TestUnloadModel
# =============================================================================


class TestUnloadModel:
    """Test POST /v1/models/{id}/unload endpoint (AC-8.3)."""

    def test_unload_model_returns_200(
        self, client: TestClient, mock_model_manager: MagicMock
    ) -> None:
        """AC-8.3: POST /v1/models/{id}/unload returns 200 on success."""
        response = client.post(f"{MODELS_ENDPOINT}/{MODEL_PHI4}/unload")

        assert response.status_code == status.HTTP_200_OK

    def test_unload_model_calls_manager(
        self, client: TestClient, mock_model_manager: MagicMock
    ) -> None:
        """AC-8.3: Unload endpoint calls model manager."""
        client.post(f"{MODELS_ENDPOINT}/{MODEL_PHI4}/unload")

        mock_model_manager.unload_model.assert_called_once_with(MODEL_PHI4)

    def test_unload_model_returns_model_info(
        self, client: TestClient, mock_model_manager: MagicMock
    ) -> None:
        """AC-8.3: Unload endpoint returns updated model info."""
        response = client.post(f"{MODELS_ENDPOINT}/{MODEL_PHI4}/unload")
        data = response.json()

        assert "id" in data
        assert "status" in data

    def test_unload_model_not_loaded_is_noop(
        self, client: TestClient, mock_model_manager: MagicMock
    ) -> None:
        """AC-8.3: Unloading non-loaded model is a no-op (returns 200)."""
        response = client.post(f"{MODELS_ENDPOINT}/{MODEL_DEEPSEEK}/unload")

        assert response.status_code == status.HTTP_200_OK


# =============================================================================
# TestUnloadModelNoManager
# =============================================================================


class TestUnloadModelNoManager:
    """Test POST /v1/models/{id}/unload when no model manager."""

    @pytest.fixture
    def app_no_manager(self) -> FastAPI:
        """Create FastAPI app without model manager."""
        from src.api.routes.models import router

        app = FastAPI()
        app.include_router(router, prefix="/v1")
        return app

    @pytest.fixture
    def client_no_manager(self, app_no_manager: FastAPI) -> TestClient:
        """Create test client without model manager."""
        return TestClient(app_no_manager)

    def test_unload_returns_503_when_no_manager(
        self, client_no_manager: TestClient
    ) -> None:
        """Returns 503 when model manager not initialized."""
        response = client_no_manager.post(f"{MODELS_ENDPOINT}/{MODEL_PHI4}/unload")

        assert response.status_code == status.HTTP_503_SERVICE_UNAVAILABLE


# =============================================================================
# TestModelInfoResponse
# =============================================================================


class TestModelInfoResponse:
    """Test ModelInfo Pydantic model."""

    def test_model_info_fields(self) -> None:
        """ModelInfo has required fields."""
        from src.api.routes.models import ModelInfo

        info = ModelInfo(
            id=MODEL_PHI4,
            status=STATUS_LOADED,
            memory_mb=8400,
            context_length=16384,
            roles=["primary", "thinker", "coder"],
        )

        assert info.id == MODEL_PHI4
        assert info.status == STATUS_LOADED
        assert info.memory_mb == 8400
        assert info.context_length == 16384
        assert "primary" in info.roles

    def test_models_list_response_fields(self) -> None:
        """ModelsListResponse has required fields."""
        from src.api.routes.models import ModelInfo, ModelsListResponse

        response = ModelsListResponse(
            data=[
                ModelInfo(
                    id=MODEL_PHI4,
                    status=STATUS_LOADED,
                    memory_mb=8400,
                    context_length=16384,
                    roles=["primary"],
                )
            ],
            config=CONFIG_D3,
            orchestration_mode="debate",
        )

        assert len(response.data) == 1
        assert response.config == CONFIG_D3
        assert response.orchestration_mode == "debate"


# =============================================================================
# TestErrorMessages
# =============================================================================


class TestErrorMessages:
    """Test error messages are clear."""

    def test_404_error_message_includes_model_id(
        self, client: TestClient, mock_model_manager: MagicMock
    ) -> None:
        """404 error includes the model ID that wasn't found."""
        from src.services.model_manager import ModelNotAvailableError

        mock_model_manager.load_model.side_effect = ModelNotAvailableError(
            "Model 'nonexistent' not available"
        )

        response = client.post(f"{MODELS_ENDPOINT}/nonexistent/load")
        data = response.json()

        assert "nonexistent" in data["detail"]

    def test_507_error_message_includes_reason(
        self, client: TestClient, mock_model_manager: MagicMock
    ) -> None:
        """507 error includes memory limit reason."""
        from src.services.model_manager import MemoryLimitExceededError

        mock_model_manager.load_model.side_effect = MemoryLimitExceededError(
            "Memory limit exceeded: 18GB > 16GB"
        )

        response = client.post(f"{MODELS_ENDPOINT}/{MODEL_DEEPSEEK}/load")
        data = response.json()

        assert "memory" in data["detail"].lower() or "limit" in data["detail"].lower()
