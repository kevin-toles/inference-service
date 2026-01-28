"""Vision API routes for image classification.

Provides /api/v1/vision/classify endpoint for VLM-based image classification.

Reference: WBS-IMG6 - Layer 2b: VLM Refinement
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from fastapi import APIRouter, HTTPException, Request, status
from pydantic import BaseModel, Field

from src.providers.moondream import (
    MoondreamProvider,
    MoondreamInferenceError,
    MoondreamImageError,
    VisionClassifyRequest,
    VisionClassifyResponse,
    BatchClassifyRequest,
    BatchClassifyResponse,
    BatchClassifyResult,
)


if TYPE_CHECKING:
    from src.providers.moondream import MoondreamProvider as MoondreamProviderType

# =============================================================================
# Router Configuration
# =============================================================================

router = APIRouter(prefix="/api/v1/vision", tags=["vision"])
logger = logging.getLogger(__name__)


# =============================================================================
# Request/Response Models (Pydantic for API)
# =============================================================================


class ImageClassifyRequest(BaseModel):
    """API request for image classification."""
    
    image_path: str | None = Field(
        default=None,
        description="Absolute path to image file to classify",
    )
    image_base64: str | None = Field(
        default=None,
        description="Base64-encoded image data (alternative to path)",
    )
    prompt: str | None = Field(
        default=None,
        description="Custom prompt (defaults to technical diagram classification)",
    )
    max_tokens: int = Field(
        default=512,
        ge=1,
        le=4096,
        description="Maximum tokens to generate",
    )
    temperature: float = Field(
        default=0.1,
        ge=0.0,
        le=2.0,
        description="Sampling temperature (0.0 for deterministic)",
    )
    resize_image: bool = Field(
        default=True,
        description="Resize large images for faster inference (recommended)",
    )

    class Config:
        json_schema_extra = {
            "example": {
                "image_path": "/Users/kevintoles/POC/ai-platform-data/books/images/sample.png",
                "max_tokens": 50,
                "temperature": 0.0,
                "resize_image": True,
            }
        }


class ImageClassifyResponse(BaseModel):
    """API response for image classification."""
    
    classification: str = Field(
        description="Classification result: 'technical' or 'discard'",
    )
    raw_response: str = Field(
        description="Full text response from VLM",
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence score 0-1 based on response clarity",
    )
    model_id: str = Field(
        description="Model used for inference",
    )
    usage: dict[str, int] = Field(
        description="Token usage statistics",
    )

    class Config:
        json_schema_extra = {
            "example": {
                "classification": "technical",
                "raw_response": "technical",
                "confidence": 0.95,
                "model_id": "deepseek-vl2-tiny",
                "usage": {
                    "prompt_tokens": 256,
                    "completion_tokens": 1,
                    "total_tokens": 257,
                },
            }
        }


class VisionHealthResponse(BaseModel):
    """Health check response for vision service."""
    
    status: str = Field(description="Service status")
    model_loaded: bool = Field(description="Whether VLM is loaded")
    model_id: str | None = Field(description="Loaded model ID")


class BatchClassifyRequestAPI(BaseModel):
    """API request for batch image classification."""
    
    image_paths: list[str] = Field(
        description="List of absolute paths to image files to classify",
    )
    prompt: str | None = Field(
        default=None,
        description="Custom prompt (defaults to technical diagram classification)",
    )
    resize_images: bool = Field(
        default=True,
        description="Resize large images for faster processing",
    )

    class Config:
        json_schema_extra = {
            "example": {
                "image_paths": [
                    "/path/to/image1.png",
                    "/path/to/image2.png",
                ],
                "resize_images": True,
            }
        }


class BatchClassifyResultAPI(BaseModel):
    """Result for a single image in batch."""
    
    image_path: str
    classification: str
    raw_response: str
    confidence: float
    error: str | None = None


class BatchClassifyResponseAPI(BaseModel):
    """API response for batch image classification."""
    
    results: list[BatchClassifyResultAPI]
    total_images: int
    successful: int
    failed: int
    model_id: str
    processing_time_seconds: float
    images_per_second: float


# =============================================================================
# Helper Functions
# =============================================================================


def _get_vision_provider(request: Request) -> MoondreamProvider | None:
    """Get vision provider from app state.

    Args:
        request: FastAPI request object

    Returns:
        MoondreamProvider instance or None if not initialized
    """
    return getattr(request.app.state, "vision_provider", None)


# =============================================================================
# Endpoints
# =============================================================================


@router.get("/health", response_model=VisionHealthResponse)
async def vision_health(request: Request) -> VisionHealthResponse:
    """Check vision service health and model status.
    
    Returns whether the VLM is loaded and ready for inference.
    """
    provider = _get_vision_provider(request)
    
    if provider is None:
        return VisionHealthResponse(
            status="unavailable",
            model_loaded=False,
            model_id=None,
        )
    
    return VisionHealthResponse(
        status="ready" if provider.is_loaded else "available",
        model_loaded=provider.is_loaded,
        model_id=provider.model_info.model_id if provider.is_loaded else None,
    )


@router.post("/classify", response_model=ImageClassifyResponse)
async def classify_image(
    request: Request,
    body: ImageClassifyRequest,
) -> ImageClassifyResponse:
    """Classify an image using the Vision Language Model.
    
    This endpoint is designed for Layer 2b refinement in the image
    classification pipeline. Use it when CLIP confidence is uncertain
    (0.3-0.7 range) to get a second opinion from the VLM.
    
    Args:
        request: FastAPI request object
        body: Image classification request
        
    Returns:
        Classification result with confidence score
        
    Raises:
        HTTPException: 400 if no image provided
        HTTPException: 404 if image not found
        HTTPException: 503 if VLM not loaded
    """
    # Validate request
    if not body.image_path and not body.image_base64:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Either image_path or image_base64 must be provided",
        )
    
    # Get provider
    provider = _get_vision_provider(request)
    if provider is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Vision provider not initialized. VLM model not configured.",
        )
    
    # Ensure model is loaded
    if not provider.is_loaded:
        logger.info("Loading VLM model on demand...")
        try:
            await provider.load()
        except Exception as e:
            logger.error(f"Failed to load VLM: {e}")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Failed to load VLM model: {str(e)}",
            )
    
    # Build request
    vision_request = VisionClassifyRequest(
        image_path=body.image_path,
        image_base64=body.image_base64,
        prompt=body.prompt,
        max_tokens=body.max_tokens,
        temperature=body.temperature,
        resize_image=body.resize_image,
    )
    
    # Run classification
    try:
        result = await provider.classify_image(vision_request)
    except MoondreamImageError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Image not found or invalid: {e.reason}",
        )
    except MoondreamInferenceError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"VLM inference failed: {e.reason}",
        )
    
    return ImageClassifyResponse(
        classification=result.classification,
        raw_response=result.raw_response,
        confidence=result.confidence,
        model_id=result.model_id,
        usage=result.usage,
    )


@router.post("/load")
async def load_vision_model(request: Request) -> dict[str, str]:
    """Explicitly load the VLM model into memory.
    
    Use this endpoint to pre-load the model before classification
    to avoid cold-start latency on the first request.
    """
    provider = _get_vision_provider(request)
    if provider is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Vision provider not configured",
        )
    
    if provider.is_loaded:
        return {"status": "already_loaded", "model_id": provider.model_info.model_id}
    
    try:
        await provider.load()
        return {"status": "loaded", "model_id": provider.model_info.model_id}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to load model: {str(e)}",
        )


@router.post("/unload")
async def unload_vision_model(request: Request) -> dict[str, str]:
    """Unload the VLM model from memory.
    
    Use this to free up VRAM/RAM when the vision service is not needed.
    """
    provider = _get_vision_provider(request)
    if provider is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Vision provider not configured",
        )
    
    if not provider.is_loaded:
        return {"status": "not_loaded"}
    
    await provider.unload()
    return {"status": "unloaded"}


@router.post("/classify/batch", response_model=BatchClassifyResponseAPI)
async def classify_batch(
    request: Request,
    body: BatchClassifyRequestAPI,
) -> BatchClassifyResponseAPI:
    """Classify multiple images in a batch.
    
    More efficient than calling /classify multiple times:
    - Model stays loaded (no per-request overhead)
    - Images optionally resized for speed
    - Continues on individual failures
    
    Args:
        request: FastAPI request object
        body: Batch classification request with image paths
        
    Returns:
        Results for each image with aggregate statistics
    """
    if not body.image_paths:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="image_paths list cannot be empty",
        )
    
    # Get provider
    provider = _get_vision_provider(request)
    if provider is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Vision provider not initialized",
        )
    
    # Ensure model is loaded
    if not provider.is_loaded:
        logger.info("Loading VLM model on demand for batch processing...")
        try:
            await provider.load()
        except Exception as e:
            logger.error(f"Failed to load VLM: {e}")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Failed to load VLM model: {str(e)}",
            )
    
    # Build batch request
    batch_request = BatchClassifyRequest(
        image_paths=body.image_paths,
        prompt=body.prompt,
        resize_images=body.resize_images,
    )
    
    # Run batch classification
    result = await provider.classify_batch(batch_request)
    
    # Calculate images per second
    ips = result.successful / result.processing_time_seconds if result.processing_time_seconds > 0 else 0
    
    return BatchClassifyResponseAPI(
        results=[
            BatchClassifyResultAPI(
                image_path=r.image_path,
                classification=r.classification,
                raw_response=r.raw_response,
                confidence=r.confidence,
                error=r.error,
            )
            for r in result.results
        ],
        total_images=result.total_images,
        successful=result.successful,
        failed=result.failed,
        model_id=result.model_id,
        processing_time_seconds=result.processing_time_seconds,
        images_per_second=round(ips, 2),
    )
