"""Health check API routes for inference-service.

Provides liveness (/health) and readiness (/health/ready) endpoints
for Kubernetes probes and service monitoring.

Reference: WBS-INF7 AC-7.1 through AC-7.4
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from fastapi import APIRouter, Request, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field


if TYPE_CHECKING:
    from src.services.model_manager import ModelManager


# =============================================================================
# Constants (S1192: Avoid duplicated string literals)
# =============================================================================

STATUS_OK = "ok"
STATUS_READY = "ready"
STATUS_NOT_READY = "not_ready"
SERVICE_NAME = "inference-service"
SERVICE_VERSION = "0.1.0"
REASON_NO_MODELS = "No models loaded"
REASON_NOT_INITIALIZED = "Model manager not initialized"


# =============================================================================
# Response Models
# =============================================================================


class HealthResponse(BaseModel):
    """Response model for /health liveness endpoint.

    AC-7.1: GET /health returns 200 when service is up.
    """

    status: str = Field(
        default=STATUS_OK,
        description="Service health status",
        examples=["ok"],
    )
    service: str = Field(
        default=SERVICE_NAME,
        description="Service name",
        examples=["inference-service"],
    )
    version: str = Field(
        default=SERVICE_VERSION,
        description="Service version",
        examples=["0.1.0"],
    )


class ReadinessResponse(BaseModel):
    """Response model for /health/ready readiness endpoint.

    AC-7.2: GET /health/ready returns 200 when models loaded.
    AC-7.3: GET /health/ready returns 503 when no models loaded.
    AC-7.4: Health response includes loaded_models list.
    """

    status: str = Field(
        description="Readiness status",
        examples=["ready", "not_ready"],
    )
    loaded_models: list[str] = Field(
        default_factory=list,
        description="List of currently loaded model IDs",
        examples=[["phi-4", "deepseek-r1-7b"]],
    )
    config: str | None = Field(
        default=None,
        description="Active configuration preset ID",
        examples=["D3"],
    )
    orchestration_mode: str | None = Field(
        default=None,
        description="Active orchestration mode",
        examples=["debate", "single"],
    )
    reason: str | None = Field(
        default=None,
        description="Reason for not ready status",
        examples=["No models loaded"],
    )
    progress: str | None = Field(
        default=None,
        description="Loading progress (e.g., '1/2')",
        examples=["1/2"],
    )


# =============================================================================
# Router
# =============================================================================

router = APIRouter(tags=["health"])


# =============================================================================
# Endpoints
# =============================================================================


@router.get(
    "/health",
    response_model=HealthResponse,
    status_code=status.HTTP_200_OK,
    summary="Liveness check",
    description="Returns 200 if the service is running. Used for K8s liveness probe.",
)
async def health_check() -> HealthResponse:
    """Liveness probe endpoint.

    AC-7.1: GET /health returns 200 when service is up.

    Returns:
        HealthResponse with status 'ok'.
    """
    return HealthResponse(
        status=STATUS_OK,
        service=SERVICE_NAME,
        version=SERVICE_VERSION,
    )


@router.get(
    "/health/ready",
    response_model=ReadinessResponse,
    responses={
        200: {"description": "Service is ready", "model": ReadinessResponse},
        503: {"description": "Service is not ready", "model": ReadinessResponse},
    },
    summary="Readiness check",
    description="Returns 200 if models are loaded. Used for K8s readiness probe.",
)
async def readiness_check(request: Request) -> JSONResponse:
    """Readiness probe endpoint.

    AC-7.2: GET /health/ready returns 200 when models loaded.
    AC-7.3: GET /health/ready returns 503 when no models loaded.
    AC-7.4: Health response includes loaded_models list.

    Args:
        request: FastAPI request to access app state.

    Returns:
        JSONResponse with readiness status and model information.
    """
    # Check if model manager is initialized
    model_manager: ModelManager | None = getattr(
        request.app.state, "model_manager", None
    )

    if model_manager is None:
        response = ReadinessResponse(
            status=STATUS_NOT_READY,
            loaded_models=[],
            reason=REASON_NOT_INITIALIZED,
        )
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content=response.model_dump(exclude_none=True),
        )

    # Get loaded models
    loaded_models = model_manager.get_loaded_models()

    # Check if any models are loaded
    if not loaded_models:
        response = ReadinessResponse(
            status=STATUS_NOT_READY,
            loaded_models=[],
            reason=REASON_NO_MODELS,
        )
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content=response.model_dump(exclude_none=True),
        )

    # Service is ready - models are loaded
    config_preset: str | None = getattr(request.app.state, "config_preset", None)
    orchestration_mode: str | None = getattr(
        request.app.state, "orchestration_mode", None
    )

    response = ReadinessResponse(
        status=STATUS_READY,
        config=config_preset,
        loaded_models=loaded_models,
        orchestration_mode=orchestration_mode,
    )

    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content=response.model_dump(exclude_none=True),
    )
