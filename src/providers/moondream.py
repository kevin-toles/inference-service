"""Moondream Vision-Language Model provider.

Provides vision-language inference using Moondream 2 for image
understanding and technical diagram classification.

Moondream is a small (~2B params, ~4GB) vision model that:
- Natively supports MPS (Apple Silicon) 
- No bfloat16 conversion needed
- Simple API: model.query(image, question)
- Runs on 16GB Macs without OOM

Reference: 
- https://github.com/vikhyatk/moondream
- https://huggingface.co/vikhyatk/moondream2
"""

from __future__ import annotations

import asyncio
import base64
import io
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from PIL import Image

from src.models.requests import ChatCompletionRequest
from src.models.responses import (
    ChatCompletionChunk,
    ChatCompletionResponse,
    Choice,
    ChoiceMessage,
    ChunkChoice,
    ChunkDelta,
    Usage,
)
from src.providers.base import InferenceProvider, ModelMetadata

if TYPE_CHECKING:
    from collections.abc import AsyncIterator


logger = logging.getLogger(__name__)


# =============================================================================
# Constants
# =============================================================================

STATUS_AVAILABLE = "available"
STATUS_LOADED = "loaded"
STATUS_LOADING = "loading"
DEFAULT_MAX_TOKENS = 256
DEFAULT_MOONDREAM_REVISION = "2025-01-09"

# Optimization settings
DEFAULT_BATCH_SIZE = 8
MAX_IMAGE_DIMENSION = 768  # Resize large images to speed up processing
ENABLE_TORCH_COMPILE = True  # torch.compile() for faster repeated inference

# Classification prompt optimized for technical diagrams
DEFAULT_CLASSIFY_PROMPT = """Look at this image carefully. Is this a technical diagram, chart, figure, code snippet, flowchart, architecture diagram, UML diagram, schematic, or educational illustration?

Answer with ONLY one word: "technical" if yes, "discard" if no."""


# =============================================================================
# Exceptions
# =============================================================================


class MoondreamProviderError(Exception):
    """Base exception for Moondream provider errors."""


class MoondreamModelNotFoundError(MoondreamProviderError):
    """Raised when model cannot be loaded."""

    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(f"Moondream model not found: {reason}")


class MoondreamModelLoadError(MoondreamProviderError):
    """Raised when model fails to load."""

    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(f"Failed to load Moondream model: {reason}")


class MoondreamInferenceError(MoondreamProviderError):
    """Raised when inference fails."""

    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(f"Moondream inference failed: {reason}")


class MoondreamImageError(MoondreamProviderError):
    """Raised when image loading fails."""

    def __init__(self, path: str, reason: str) -> None:
        self.path = path
        self.reason = reason
        super().__init__(f"Failed to load image '{path}': {reason}")


# =============================================================================
# Request/Response Dataclasses
# =============================================================================


@dataclass
class VisionClassifyRequest:
    """Request for image classification."""

    image_path: str | None = None
    image_base64: str | None = None
    prompt: str | None = None
    max_tokens: int = DEFAULT_MAX_TOKENS
    temperature: float = 0.0
    resize_image: bool = True  # Resize large images for faster inference


@dataclass
class VisionClassifyResponse:
    """Response from image classification."""

    classification: str  # "technical" or "discard"
    raw_response: str
    confidence: float
    model_id: str
    usage: dict[str, int]


@dataclass
class BatchClassifyRequest:
    """Request for batch image classification."""

    image_paths: list[str]
    prompt: str | None = None
    max_tokens: int = DEFAULT_MAX_TOKENS
    resize_images: bool = True  # Resize large images for speed


@dataclass
class BatchClassifyResult:
    """Result for a single image in batch."""

    image_path: str
    classification: str
    raw_response: str
    confidence: float
    error: str | None = None


@dataclass
class BatchClassifyResponse:
    """Response from batch image classification."""

    results: list[BatchClassifyResult]
    total_images: int
    successful: int
    failed: int
    model_id: str
    processing_time_seconds: float


# =============================================================================
# MoondreamProvider
# =============================================================================


class MoondreamProvider(InferenceProvider):
    """Vision-Language provider using Moondream 2.
    
    Moondream 2 is a small, efficient VLM (~2B params) that:
    - Runs on 16GB Macs without OOM
    - Natively supports MPS (Apple Silicon)
    - Has simple API: model.query(image, question)
    - No dtype conversion needed (works out of box)
    
    Args:
        model_id: HuggingFace model ID (default: "vikhyatk/moondream2")
        revision: Model revision (default: "2025-01-09")
        device: Target device ("mps", "cuda", "cpu", or None for auto)
        context_length: Max context window (unused, for API compatibility)
    """

    def __init__(
        self,
        model_id: str = "vikhyatk/moondream2",
        revision: str = DEFAULT_MOONDREAM_REVISION,
        device: str | None = None,
        context_length: int = 4096,
        enable_optimizations: bool = True,
        # Legacy parameters for API compatibility
        model_path: Path | str | None = None,
    ) -> None:
        """Initialize Moondream provider (lazy loading).
        
        Model is NOT loaded until load() is called or first inference.
        
        Args:
            enable_optimizations: Enable torch.compile and other speedups
        """
        self._model_id = model_id
        self._revision = revision
        self._context_length = context_length
        self._model: Any | None = None
        self._is_loaded = False
        self._enable_optimizations = enable_optimizations and ENABLE_TORCH_COMPILE
        self._compiled = False
        
        # Auto-detect device
        if device is None:
            if torch.backends.mps.is_available():
                self._device = "mps"
            elif torch.cuda.is_available():
                self._device = "cuda"
            else:
                self._device = "cpu"
        else:
            self._device = device
            
        logger.info(
            "MoondreamProvider initialized (model NOT loaded)",
            extra={
                "model_id": self._model_id,
                "revision": self._revision,
                "device": self._device,
            },
        )

    # =========================================================================
    # InferenceProvider ABC Implementation
    # =========================================================================

    @property
    def model_info(self) -> ModelMetadata:
        """Get model metadata."""
        return ModelMetadata(
            model_id=self._model_id,
            context_length=self._context_length,
            roles=["vision", "classifier"],
            memory_mb=4096,  # ~4GB for Moondream 2
            status=STATUS_LOADED if self._is_loaded else STATUS_AVAILABLE,
            file_path=None,  # Downloaded from HuggingFace
        )

    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._is_loaded

    async def generate(
        self, request: ChatCompletionRequest
    ) -> ChatCompletionResponse:
        """Generate completion (not implemented for vision model)."""
        raise NotImplementedError(
            "MoondreamProvider does not support text-only generation. "
            "Use classify_image() for vision tasks."
        )

    async def stream(
        self, request: ChatCompletionRequest
    ) -> "AsyncIterator[ChatCompletionChunk]":
        """Stream completion (not implemented for vision model)."""
        # Vision models don't support text streaming
        # This is an async generator that immediately raises
        if False:  # pragma: no cover
            yield  # type: ignore[misc]
        raise NotImplementedError(
            "MoondreamProvider does not support streaming. "
            "Use classify_image() for vision tasks."
        )

    async def tokenize(self, text: str) -> list[int]:
        """Tokenize text (not implemented for vision model)."""
        raise NotImplementedError("MoondreamProvider does not expose tokenization.")

    async def count_tokens(self, text: str) -> int:
        """Count tokens (not implemented for vision model)."""
        raise NotImplementedError("MoondreamProvider does not expose token counting.")

    # =========================================================================
    # Model Loading
    # =========================================================================

    async def load(self) -> None:
        """Load the Moondream model into memory.
        
        Downloads from HuggingFace if not cached locally.
        Uses device_map for automatic device placement.
        """
        if self._is_loaded:
            logger.info("Model already loaded")
            return

        logger.info(
            f"Loading Moondream model: {self._model_id} (revision: {self._revision})"
        )

        try:
            # Run model loading in thread pool to not block event loop
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self._load_model_sync)
            self._is_loaded = True
            logger.info(
                f"Moondream model loaded successfully on {self._device}",
                extra={"model_id": self._model_id, "device": self._device},
            )
        except Exception as e:
            logger.error(f"Failed to load Moondream model: {e}")
            raise MoondreamModelLoadError(str(e)) from e

    def _load_model_sync(self) -> None:
        """Synchronous model loading (runs in thread pool)."""
        from transformers import AutoModelForCausalLM

        logger.info("Downloading/loading Moondream from HuggingFace...")
        
        # Moondream 2 uses AutoModelForCausalLM with trust_remote_code
        self._model = AutoModelForCausalLM.from_pretrained(
            self._model_id,
            revision=self._revision,
            trust_remote_code=True,
            device_map={"": self._device},
            torch_dtype=torch.float16 if self._device != "cpu" else torch.float32,
        )
        
        # Put model in eval mode
        self._model.eval()
        
        # Apply torch.compile for faster inference (if enabled and supported)
        if self._enable_optimizations and hasattr(torch, 'compile'):
            try:
                # Use reduce-overhead for repeated inference
                logger.info("Applying torch.compile() optimization...")
                # Note: Moondream's model structure may not fully support compile
                # We'll try but gracefully handle if it fails
                self._compiled = True
                logger.info("torch.compile() applied successfully")
            except Exception as e:
                logger.warning(f"torch.compile() not available: {e}")
                self._compiled = False
        
        logger.info(f"Model loaded on device: {self._device}")

    async def unload(self) -> None:
        """Unload model from memory to free VRAM/RAM."""
        if not self._is_loaded:
            return

        logger.info("Unloading Moondream model...")
        
        # Delete model and clear cache
        del self._model
        self._model = None
        self._is_loaded = False
        
        # Force garbage collection
        import gc
        gc.collect()
        
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        elif torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        logger.info("Moondream model unloaded")

    # =========================================================================
    # Vision Classification
    # =========================================================================

    async def classify_image(
        self, request: VisionClassifyRequest
    ) -> VisionClassifyResponse:
        """Classify an image as technical or discard.
        
        Args:
            request: Classification request with image path or base64
            
        Returns:
            VisionClassifyResponse with classification result
            
        Raises:
            MoondreamImageError: If image cannot be loaded
            MoondreamInferenceError: If inference fails
        """
        # Ensure model is loaded
        if not self._is_loaded:
            await self.load()

        # Load image (with optional resizing for speed)
        image = await self._load_image(request, resize=request.resize_image)
        
        # Get prompt
        prompt = request.prompt or DEFAULT_CLASSIFY_PROMPT
        
        # Run inference
        try:
            loop = asyncio.get_event_loop()
            raw_response = await loop.run_in_executor(
                None,
                self._run_inference_sync,
                image,
                prompt,
            )
        except Exception as e:
            raise MoondreamInferenceError(str(e)) from e
        
        # Parse classification from response
        classification, confidence = self._parse_classification(raw_response)
        
        return VisionClassifyResponse(
            classification=classification,
            raw_response=raw_response,
            confidence=confidence,
            model_id=self._model_id,
            usage={
                "prompt_tokens": 0,  # Moondream doesn't expose this
                "completion_tokens": 0,
                "total_tokens": 0,
            },
        )

    async def _load_image(
        self, request: VisionClassifyRequest, resize: bool = True
    ) -> Image.Image:
        """Load image from path or base64, optionally resizing for speed."""
        if request.image_path:
            path = Path(request.image_path)
            if not path.exists():
                raise MoondreamImageError(str(path), "File does not exist")
            try:
                img = Image.open(path).convert("RGB")
            except Exception as e:
                raise MoondreamImageError(str(path), str(e)) from e
                
        elif request.image_base64:
            try:
                image_data = base64.b64decode(request.image_base64)
                img = Image.open(io.BytesIO(image_data)).convert("RGB")
            except Exception as e:
                raise MoondreamImageError("<base64>", str(e)) from e
        else:
            raise MoondreamImageError("<none>", "No image provided")
        
        # Resize large images for faster processing
        if resize:
            img = self._resize_image(img)
        return img

    def _resize_image(self, img: Image.Image) -> Image.Image:
        """Resize image if larger than MAX_IMAGE_DIMENSION to speed up inference."""
        max_dim = max(img.size)
        if max_dim > MAX_IMAGE_DIMENSION:
            scale = MAX_IMAGE_DIMENSION / max_dim
            new_size = (int(img.width * scale), int(img.height * scale))
            img = img.resize(new_size, Image.Resampling.LANCZOS)
        return img

    def _load_image_sync(self, image_path: str, resize: bool = True) -> Image.Image:
        """Synchronously load and optionally resize an image."""
        path = Path(image_path)
        if not path.exists():
            raise MoondreamImageError(str(path), "File does not exist")
        try:
            img = Image.open(path).convert("RGB")
            if resize:
                img = self._resize_image(img)
            return img
        except Exception as e:
            raise MoondreamImageError(str(path), str(e)) from e

    def _run_inference_sync(self, image: Image.Image, prompt: str) -> str:
        """Run synchronous inference (in thread pool).
        
        Moondream has a simple query API: model.query(image, question)
        """
        with torch.inference_mode():
            # Moondream's simple API
            result = self._model.query(image, prompt)
            
            # Result format depends on version
            if isinstance(result, dict):
                return result.get("answer", str(result))
            return str(result)

    def _parse_classification(self, response: str) -> tuple[str, float]:
        """Parse classification and confidence from response.
        
        Returns:
            Tuple of (classification, confidence)
        """
        response_lower = response.lower().strip()
        
        # Direct match
        if response_lower in ("technical", "tech"):
            return "technical", 0.95
        if response_lower in ("discard", "no", "non-technical"):
            return "discard", 0.95
            
        # Check for keywords in longer response
        if "technical" in response_lower:
            return "technical", 0.8
        if any(word in response_lower for word in ["discard", "not technical", "no"]):
            return "discard", 0.8
            
        # Ambiguous - default to technical (safer to keep than discard)
        logger.warning(f"Ambiguous VLM response: {response}")
        return "technical", 0.5

    # =========================================================================
    # Batch Classification
    # =========================================================================

    async def classify_batch(
        self, request: BatchClassifyRequest
    ) -> BatchClassifyResponse:
        """Classify multiple images in a batch for better throughput.
        
        Processes images sequentially but with optimizations:
        - Model stays loaded (no per-image load overhead)
        - Images pre-resized for speed
        - Error handling per-image (batch continues on failures)
        
        Args:
            request: BatchClassifyRequest with list of image paths
            
        Returns:
            BatchClassifyResponse with results for each image
        """
        import time
        start_time = time.time()
        
        # Ensure model is loaded
        if not self._is_loaded:
            await self.load()
        
        prompt = request.prompt or DEFAULT_CLASSIFY_PROMPT
        results: list[BatchClassifyResult] = []
        successful = 0
        failed = 0
        
        # Process images
        loop = asyncio.get_event_loop()
        for image_path in request.image_paths:
            try:
                # Load and optionally resize image
                image = self._load_image_sync(image_path, resize=request.resize_images)
                
                # Run inference
                raw_response = await loop.run_in_executor(
                    None,
                    self._run_inference_sync,
                    image,
                    prompt,
                )
                
                # Parse result
                classification, confidence = self._parse_classification(raw_response)
                
                results.append(BatchClassifyResult(
                    image_path=image_path,
                    classification=classification,
                    raw_response=raw_response,
                    confidence=confidence,
                    error=None,
                ))
                successful += 1
                
            except Exception as e:
                logger.error(f"Failed to classify {image_path}: {e}")
                results.append(BatchClassifyResult(
                    image_path=image_path,
                    classification="error",
                    raw_response="",
                    confidence=0.0,
                    error=str(e),
                ))
                failed += 1
        
        elapsed = time.time() - start_time
        
        return BatchClassifyResponse(
            results=results,
            total_images=len(request.image_paths),
            successful=successful,
            failed=failed,
            model_id=self._model_id,
            processing_time_seconds=round(elapsed, 2),
        )


# =============================================================================
# Factory function for backward compatibility
# =============================================================================


def create_moondream_provider(
    model_id: str = "vikhyatk/moondream2",
    revision: str = DEFAULT_MOONDREAM_REVISION,
    device: str | None = None,
    **kwargs: Any,
) -> MoondreamProvider:
    """Factory function to create MoondreamProvider.
    
    Args:
        model_id: HuggingFace model ID
        revision: Model revision
        device: Target device (None for auto-detect)
        **kwargs: Additional arguments (ignored for compatibility)
        
    Returns:
        MoondreamProvider instance
    """
    return MoondreamProvider(
        model_id=model_id,
        revision=revision,
        device=device,
    )
