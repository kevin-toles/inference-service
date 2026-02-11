"""Models API routes for listing, loading, and unloading models.

Provides endpoints for model lifecycle management.

Reference: WBS-INF8 AC-8.1 through AC-8.4
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from fastapi import APIRouter, HTTPException, Request, status
from pydantic import BaseModel


if TYPE_CHECKING:
    from src.services.model_manager import ModelManager

# =============================================================================
# Router Configuration
# =============================================================================

router = APIRouter(tags=["models"])


# =============================================================================
# Response Models
# =============================================================================


class ModelInfo(BaseModel):
    """Information about a single model.

    Attributes:
        id: Model identifier (e.g., "phi-4")
        status: Current status ("loaded" or "available")
        memory_mb: Memory usage in megabytes
        context_length: Maximum context window size
        roles: List of roles the model can fulfill
    """

    id: str
    status: str
    memory_mb: int
    context_length: int
    roles: list[str]


class ModelsListResponse(BaseModel):
    """Response for GET /v1/models.

    Attributes:
        data: List of model information objects
        config: Current preset configuration ID
        orchestration_mode: Current orchestration mode (e.g., "debate")
    """

    data: list[ModelInfo]
    config: str
    orchestration_mode: str


class ModelActionResponse(BaseModel):
    """Response for load/unload actions.

    Attributes:
        id: Model identifier
        status: Updated status after action
        message: Human-readable status message
    """

    id: str
    status: str
    message: str


class ModelConfigUpdateRequest(BaseModel):
    """Request body for PATCH /v1/models/{model_id}/config.

    Attributes:
        context_length: New context window size (optional)
        gpu_layers: New GPU layer count (optional; -1 = all, 0 = CPU)
    """

    context_length: int | None = None
    gpu_layers: int | None = None


class ModelConfigUpdateResponse(BaseModel):
    """Response for PATCH /v1/models/{model_id}/config.

    Attributes:
        model_id: Model identifier
        updated_fields: List of field names that were actually changed
        status: Always "ok"
    """

    model_id: str
    updated_fields: list[str]
    status: str = "ok"


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
    manager: ModelManager | None = getattr(request.app.state, "model_manager", None)
    if manager is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model manager not initialized",
        )
    return manager


def _model_info_to_response(model_info: object) -> ModelInfo:
    """Convert ModelInfo dataclass to response model.

    Args:
        model_info: Model info from model manager

    Returns:
        ModelInfo response object
    """
    return ModelInfo(
        id=model_info.model_id,  # type: ignore[attr-defined]
        status=model_info.status,  # type: ignore[attr-defined]
        memory_mb=int(model_info.size_gb * 1000),  # type: ignore[attr-defined]
        context_length=model_info.context_length,  # type: ignore[attr-defined]
        roles=model_info.roles,  # type: ignore[attr-defined]
    )


# =============================================================================
# Endpoints
# =============================================================================


@router.get(
    "/models",
    response_model=ModelsListResponse,
    summary="List all models",
    description="Returns a list of all available and loaded models with their status.",
)
async def list_models(request: Request) -> ModelsListResponse:
    """List all available and loaded models.

    AC-8.1: GET /v1/models lists available and loaded models.
    AC-8.4: Model response includes status, memory, context, roles.

    Args:
        request: FastAPI request object

    Returns:
        ModelsListResponse with model data and configuration
    """
    manager = _get_model_manager(request)

    # Get all models from manager
    all_models = manager.list_all_models()

    # Convert to response format
    data = [_model_info_to_response(m) for m in all_models]

    # Get config from app state
    config = getattr(request.app.state, "config_preset", "unknown")
    orchestration_mode = getattr(request.app.state, "orchestration_mode", "unknown")

    return ModelsListResponse(
        data=data,
        config=config,
        orchestration_mode=orchestration_mode,
    )


@router.post(
    "/models/{model_id}/load",
    response_model=ModelActionResponse,
    summary="Load a model",
    description="Load a specific model into memory for inference.",
    responses={
        404: {"description": "Model not found"},
        507: {"description": "Insufficient memory to load model"},
    },
)
async def load_model(request: Request, model_id: str) -> ModelActionResponse:
    """Load a model by ID.

    AC-8.2: POST /v1/models/{id}/load loads specified model.

    Args:
        request: FastAPI request object
        model_id: ID of the model to load

    Returns:
        ModelActionResponse with updated status

    Raises:
        HTTPException: 404 if model not found, 507 if memory exceeded
    """
    # Import exceptions here to avoid circular imports
    from src.services.model_manager import (
        MemoryLimitExceededError,
        ModelNotAvailableError,
    )

    manager = _get_model_manager(request)

    try:
        await manager.load_model(model_id)
        return ModelActionResponse(
            id=model_id,
            status="loaded",
            message=f"Model '{model_id}' loaded successfully",
        )
    except ModelNotAvailableError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        ) from e
    except MemoryLimitExceededError as e:
        raise HTTPException(
            status_code=status.HTTP_507_INSUFFICIENT_STORAGE,
            detail=str(e),
        ) from e


@router.post(
    "/models/{model_id}/unload",
    response_model=ModelActionResponse,
    summary="Unload a model",
    description="Unload a model from memory to free resources.",
)
async def unload_model(request: Request, model_id: str) -> ModelActionResponse:
    """Unload a model by ID.

    AC-8.3: POST /v1/models/{id}/unload unloads specified model.

    Args:
        request: FastAPI request object
        model_id: ID of the model to unload

    Returns:
        ModelActionResponse with updated status
    """
    manager = _get_model_manager(request)

    # Unload is always successful (no-op if not loaded)
    await manager.unload_model(model_id)

    return ModelActionResponse(
        id=model_id,
        status="available",
        message=f"Model '{model_id}' unloaded successfully",
    )


@router.patch(
    "/models/{model_id}/config",
    response_model=ModelConfigUpdateResponse,
    summary="Update model configuration",
    description="Update runtime configuration for a model (e.g., context_length, gpu_layers).",
    responses={
        404: {"description": "Model not found in configuration"},
    },
)
async def update_model_config(
    request: Request,
    model_id: str,
    body: ModelConfigUpdateRequest,
) -> ModelConfigUpdateResponse:
    """Update model runtime config and log changes to audit trail.

    LLM Operations Mesh — Phase C (AC-C.1, AC-C.2).

    Note: Changes take effect on next model load. If the model is already
    loaded, it will use the old config until reloaded.

    Args:
        request: FastAPI request object
        model_id: ID of the model to configure
        body: Fields to update

    Returns:
        ModelConfigUpdateResponse with list of changed fields

    Raises:
        HTTPException: 404 if model not found
    """
    from src.services.model_manager import ModelNotAvailableError

    manager = _get_model_manager(request)

    # Build kwargs from non-None fields
    updates = {k: v for k, v in body.model_dump().items() if v is not None}

    try:
        changed = await manager.update_model_config(model_id, **updates)
    except ModelNotAvailableError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        ) from e

    return ModelConfigUpdateResponse(
        model_id=model_id,
        updated_fields=list(changed.keys()),
    )
