"""DeepSeek VL2 Vision-Language Model provider.

Provides vision-language inference using DeepSeek-VL2-Tiny for image
understanding and technical diagram classification.

This provider differs from LlamaCpp in that:
- Uses HuggingFace transformers (not GGUF)
- Accepts image input (path or base64)
- Returns structured classification results

Patterns applied:
- InferenceProvider ABC implementation (adapted for vision)
- Async context manager for resource management
- Exception classes ending in "Error" (AP-7)
- PEP 604 union syntax (X | None)

Reference: WBS-IMG6 - Layer 2b: VLM Refinement
"""

from __future__ import annotations

import asyncio
import base64
import logging
from collections.abc import AsyncIterator
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
    from src.providers.backends import DeviceBackend


logger = logging.getLogger(__name__)


# =============================================================================
# Constants
# =============================================================================

STATUS_AVAILABLE = "available"
STATUS_LOADED = "loaded"
STATUS_LOADING = "loading"
DEFAULT_CONTEXT_LENGTH = 1024  # Minimum viable for MPS - prevents OOM
DEFAULT_MAX_NEW_TOKENS = 64    # Minimum for classification (just need yes/no)
MAX_IMAGE_SIZE = 384           # Resize large images to reduce VRAM


# =============================================================================
# Exceptions (AP-7: Exception names end in "Error")
# =============================================================================


class DeepSeekVLProviderError(Exception):
    """Base exception for DeepSeek VL provider errors."""


class DeepSeekVLModelNotFoundError(DeepSeekVLProviderError):
    """Raised when model directory is not found."""

    def __init__(self, model_path: str) -> None:
        self.model_path = model_path
        super().__init__(f"DeepSeek VL model not found at: {model_path}")


class DeepSeekVLModelLoadError(DeepSeekVLProviderError):
    """Raised when model fails to load."""

    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(f"Failed to load DeepSeek VL model: {reason}")


class DeepSeekVLInferenceError(DeepSeekVLProviderError):
    """Raised when inference fails."""

    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(f"DeepSeek VL inference failed: {reason}")


class DeepSeekVLImageError(DeepSeekVLProviderError):
    """Raised when image loading fails."""

    def __init__(self, path: str, reason: str) -> None:
        self.path = path
        self.reason = reason
        super().__init__(f"Failed to load image {path}: {reason}")


# =============================================================================
# Vision Request/Response Models
# =============================================================================


@dataclass
class VisionClassifyRequest:
    """Request for image classification via VLM.
    
    Attributes:
        image_path: Path to image file to classify.
        image_base64: Base64-encoded image data (alternative to path).
        prompt: Custom prompt (defaults to diagram classification prompt).
        max_tokens: Maximum tokens to generate.
        temperature: Sampling temperature (0.0 for deterministic).
    """
    image_path: str | None = None
    image_base64: str | None = None
    prompt: str | None = None
    max_tokens: int = DEFAULT_MAX_NEW_TOKENS
    temperature: float = 0.1


@dataclass
class VisionClassifyResponse:
    """Response from VLM image classification.
    
    Attributes:
        classification: The classification result ("technical" or "discard").
        raw_response: Full text response from VLM.
        confidence: Estimated confidence (based on response clarity).
        model_id: Model used for inference.
        usage: Token usage statistics.
    """
    classification: str
    raw_response: str
    confidence: float
    model_id: str
    usage: dict[str, int]


# =============================================================================
# Default Prompts
# =============================================================================

DIAGRAM_CLASSIFICATION_PROMPT = """Look at this image and answer with ONLY one word: "technical" or "discard"

Classify as "technical" if the image shows ANY of:
- Software architecture diagram
- UML diagram (class, sequence, state, activity, component, deployment)
- Flowchart or process diagram
- Database schema or ER diagram
- Network topology diagram
- System design diagram
- Code screenshot or code listing
- API documentation diagram
- State machine diagram

Classify as "discard" if the image shows:
- Photograph of people, objects, or scenery
- Decorative illustration or artwork
- Company logo or brand graphic
- Generic marketing image
- Book cover or title page
- Author photo
- Stock photography

Your answer (one word only):"""


# =============================================================================
# DeepSeekVLProvider Implementation
# =============================================================================


class DeepSeekVLProvider(InferenceProvider):
    """DeepSeek VL2 vision-language provider.

    Uses DeepSeek-VL2-Tiny for image understanding with Metal acceleration
    on Mac. Designed for technical diagram classification in the image
    classification pipeline (Layer 2b).

    Args:
        model_path: Path to model directory containing safetensors.
        model_id: Unique model identifier.
        context_length: Maximum context window in tokens.
        device: Device to use ("mps", "cuda", "cpu", or None for auto).

    Raises:
        DeepSeekVLModelNotFoundError: If model directory does not exist.

    Example:
        >>> provider = DeepSeekVLProvider(
        ...     model_path=Path("/models/deepseek-vl2-tiny"),
        ...     model_id="deepseek-vl2-tiny",
        ... )
        >>> async with provider:
        ...     result = await provider.classify_image(request)
    """

    def __init__(
        self,
        model_path: Path,
        model_id: str,
        context_length: int = DEFAULT_CONTEXT_LENGTH,
        device: str | None = None,
        backend: "DeviceBackend | None" = None,
    ) -> None:
        """Initialize DeepSeek VL provider.
        
        Args:
            model_path: Path to model directory containing safetensors.
            model_id: Unique model identifier.
            context_length: Maximum context window in tokens.
            device: Device to use ("mps", "cuda", "cpu", or None for auto).
            backend: Optional DeviceBackend for dtype handling. If None,
                    created via DeviceBackendFactory.create_backend(device).
        """
        self._model_path = Path(model_path)
        self._model_id = model_id
        self._context_length = context_length
        self._device = self._select_device(device)
        self._status = STATUS_AVAILABLE
        
        # Create or use provided backend for dtype handling
        if backend is not None:
            self._backend = backend
        else:
            from .backends import DeviceBackendFactory, DeviceBackendFactoryError
            try:
                self._backend = DeviceBackendFactory.create_backend(self._device)
            except DeviceBackendFactoryError:
                # Device not in registry (e.g., cpu, cuda without backend)
                # Use None and skip dtype conversion
                self._backend = None
        
        # Model components (loaded lazily)
        self._model: Any = None
        self._processor: Any = None
        self._tokenizer: Any = None
        
        # Validate model path exists
        if not self._model_path.exists():
            raise DeepSeekVLModelNotFoundError(str(self._model_path))

    def _select_device(self, device: str | None) -> str:
        """Select the best available device."""
        if device is not None:
            return device
        
        # Auto-detect best device
        if torch.backends.mps.is_available():
            return "mps"
        elif torch.cuda.is_available():
            return "cuda"
        return "cpu"

    @property
    def model_info(self) -> ModelMetadata:
        """Get model metadata."""
        return ModelMetadata(
            model_id=self._model_id,
            context_length=self._context_length,
            roles=["vision"],
            memory_mb=3200,  # ~3.2GB
            status=self._status,
            file_path=str(self._model_path),
        )

    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._model is not None and self._processor is not None

    async def load(self) -> None:
        """Load the VLM model into memory."""
        if self.is_loaded:
            logger.info("DeepSeek VL model already loaded")
            return

        self._status = STATUS_LOADING
        logger.info(
            "Loading DeepSeek VL model",
            extra={"model_id": self._model_id, "device": self._device},
        )

        try:
            # Run model loading in executor to not block event loop
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self._load_model_sync)
            self._status = STATUS_LOADED
            logger.info(
                "DeepSeek VL model loaded successfully",
                extra={"model_id": self._model_id, "device": self._device},
            )
        except Exception as e:
            self._status = STATUS_AVAILABLE
            raise DeepSeekVLModelLoadError(str(e)) from e

    def _load_model_sync(self) -> None:
        """Synchronous model loading (runs in executor).
        
        Loading sequence (per WBS-VLM5):
        0. Apply SigLIP attention patch for MPS (removes xformers dependency)
        1. Load model via from_pretrained()
        2. Apply dtype conversion via backend.ensure_dtype_compatibility()
        3. Move model to device via .to(device)
        """
        # Step 0: Apply SigLIP attention patch for non-CUDA devices
        # This must happen BEFORE importing DeepSeek-VL2 models
        # xformers only works on CUDA, so we need the patch for MPS and CPU
        from .backends import apply_siglip_attention_patch
        
        device_lower = str(self._device).lower() if self._device else ""
        needs_patch = "mps" in device_lower or "cpu" in device_lower
        
        if needs_patch:
            if apply_siglip_attention_patch():
                logger.info(
                    "Applied SigLIP attention patch (xformers replacement)",
                    extra={"device": self._device},
                )
            else:
                logger.warning(
                    "Failed to apply SigLIP attention patch - xformers may be required",
                    extra={"device": self._device},
                )
        
        try:
            from transformers import AutoModelForCausalLM
            from deepseek_vl2.models import DeepseekVLV2Processor, DeepseekVLV2ForCausalLM
        except ImportError as e:
            raise DeepSeekVLModelLoadError(
                f"Required packages not installed: {e}. "
                "Install with: pip install deepseek-vl2"
            ) from e

        # Load processor and tokenizer
        self._processor = DeepseekVLV2Processor.from_pretrained(str(self._model_path))
        self._tokenizer = self._processor.tokenizer

        # Step 1: Load model via from_pretrained()
        # Use eager attention to avoid flash attention compatibility issues
        logger.info(
            "Loading model via from_pretrained()",
            extra={"model_id": self._model_id, "device": self._device},
        )
        self._model = AutoModelForCausalLM.from_pretrained(
            str(self._model_path),
            trust_remote_code=True,
            attn_implementation="eager",
            low_cpu_mem_usage=True,
        )

        # Step 2: Apply dtype conversion via backend (BEFORE .to(device))
        # This handles bfloat16 -> float32 conversion on MPS
        if self._backend is not None:
            logger.info(
                "Applying dtype compatibility via backend",
                extra={"backend": type(self._backend).__name__},
            )
            self._model = self._backend.ensure_dtype_compatibility(self._model)

        # Step 3: Move model to device and set to eval mode
        logger.info(
            "Moving model to device",
            extra={"device": self._device},
        )
        self._model = self._model.to(self._device).eval()

    async def unload(self) -> None:
        """Unload model from memory."""
        if not self.is_loaded:
            return

        logger.info("Unloading DeepSeek VL model", extra={"model_id": self._model_id})
        
        # Clear model references
        del self._model
        del self._processor
        del self._tokenizer
        self._model = None
        self._processor = None
        self._tokenizer = None
        
        # Force garbage collection
        import gc
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        elif torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        self._status = STATUS_AVAILABLE
        logger.info("DeepSeek VL model unloaded")

    async def __aenter__(self) -> "DeepSeekVLProvider":
        """Async context manager entry - load model."""
        await self.load()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit - unload model."""
        await self.unload()

    # =========================================================================
    # Vision-specific methods
    # =========================================================================

    async def classify_image(
        self,
        request: VisionClassifyRequest,
    ) -> VisionClassifyResponse:
        """Classify an image using the VLM.

        Args:
            request: Vision classification request with image and prompt.

        Returns:
            VisionClassifyResponse with classification result.

        Raises:
            DeepSeekVLInferenceError: If model not loaded or inference fails.
            DeepSeekVLImageError: If image cannot be loaded.
        """
        if not self.is_loaded:
            raise DeepSeekVLInferenceError("Model not loaded. Call load() first.")

        # Load image
        try:
            image = await self._load_image(request)
        except Exception as e:
            path = request.image_path or "<base64>"
            raise DeepSeekVLImageError(path, str(e)) from e

        # Build prompt
        prompt = request.prompt or DIAGRAM_CLASSIFICATION_PROMPT

        # Run inference in executor
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            self._run_inference_sync,
            image,
            prompt,
            request.max_tokens,
            request.temperature,
        )

        return result

    async def _load_image(self, request: VisionClassifyRequest) -> Image.Image:
        """Load image from path or base64, resize to save memory."""
        if request.image_path:
            path = Path(request.image_path)
            if not path.exists():
                raise FileNotFoundError(f"Image not found: {path}")
            img = Image.open(path).convert("RGB")
        elif request.image_base64:
            import io
            image_data = base64.b64decode(request.image_base64)
            img = Image.open(io.BytesIO(image_data)).convert("RGB")
        else:
            raise ValueError("Either image_path or image_base64 must be provided")
        
        # MEMORY OPTIMIZATION: Resize large images to reduce VRAM usage
        max_dim = max(img.size)
        if max_dim > MAX_IMAGE_SIZE:
            scale = MAX_IMAGE_SIZE / max_dim
            new_size = (int(img.size[0] * scale), int(img.size[1] * scale))
            img = img.resize(new_size, Image.Resampling.LANCZOS)
        return img

    def _run_inference_sync(
        self,
        image: Image.Image,
        prompt: str,
        max_tokens: int,
        temperature: float,
    ) -> VisionClassifyResponse:
        """Synchronous inference (runs in executor)."""
        from deepseek_vl2.utils.io import load_pil_images

        # Prepare conversation format
        conversation = [
            {
                "role": "<|User|>",
                "content": f"<image>\n{prompt}",
                "images": [image],
            },
            {"role": "<|Assistant|>", "content": ""},
        ]

        # Process inputs
        prepare_inputs = self._processor(
            conversations=conversation,
            images=[image],
            force_batchify=True,
            system_prompt="",
        ).to(self._model.device)
        
        # Convert input tensors to match model dtype (float16 on MPS, float32 on CPU)
        if self._backend is not None:
            target_dtype = self._backend.get_optimal_dtype()
            for key in prepare_inputs.keys():
                tensor = getattr(prepare_inputs, key, None)
                if tensor is not None and hasattr(tensor, 'dtype'):
                    # Convert any float type to target dtype for consistency
                    if tensor.dtype in (torch.bfloat16, torch.float32, torch.float16):
                        if tensor.dtype != target_dtype:
                            setattr(prepare_inputs, key, tensor.to(target_dtype))

        # Get image embeddings
        inputs_embeds = self._model.prepare_inputs_embeds(**prepare_inputs)

        # Generate response
        # Note: DeepseekVLV2ForCausalLM uses `language` attribute, not `language_model`
        with torch.no_grad():
            outputs = self._model.language.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=prepare_inputs.attention_mask,
                pad_token_id=self._tokenizer.eos_token_id,
                bos_token_id=self._tokenizer.bos_token_id,
                eos_token_id=self._tokenizer.eos_token_id,
                max_new_tokens=max_tokens,
                do_sample=False,  # Greedy decoding - faster, deterministic
                use_cache=False,  # MEMORY: Disable KV cache to prevent OOM
            )

        # Decode response
        raw_response = self._tokenizer.decode(
            outputs[0].cpu().tolist(),
            skip_special_tokens=True,
        ).strip()

        # Parse classification from response
        classification, confidence = self._parse_classification(raw_response)

        # Estimate token usage
        input_tokens = len(prepare_inputs.input_ids[0])
        output_tokens = len(outputs[0]) - input_tokens
        
        # CRITICAL: Explicit memory cleanup to prevent MPS OOM crashes
        del outputs, inputs_embeds, prepare_inputs, image
        if self._device == "mps":
            import gc
            gc.collect()
            torch.mps.empty_cache()

        return VisionClassifyResponse(
            classification=classification,
            raw_response=raw_response,
            confidence=confidence,
            model_id=self._model_id,
            usage={
                "prompt_tokens": input_tokens,
                "completion_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens,
            },
        )

    def _parse_classification(self, response: str) -> tuple[str, float]:
        """Parse classification from VLM response.

        Returns:
            Tuple of (classification, confidence).
            Classification is "technical" or "discard".
            Confidence is 0.0-1.0 based on response clarity.
        """
        response_lower = response.lower().strip()
        
        # Check for clear single-word response
        if response_lower == "technical":
            return "technical", 0.95
        elif response_lower == "discard":
            return "discard", 0.95
        
        # Check if response contains the keywords
        if "technical" in response_lower and "discard" not in response_lower:
            return "technical", 0.75
        elif "discard" in response_lower and "technical" not in response_lower:
            return "discard", 0.75
        
        # Ambiguous - check for related keywords
        technical_keywords = ["diagram", "architecture", "uml", "flowchart", "code", "schema"]
        discard_keywords = ["photo", "logo", "decorative", "person", "cover"]
        
        tech_count = sum(1 for kw in technical_keywords if kw in response_lower)
        disc_count = sum(1 for kw in discard_keywords if kw in response_lower)
        
        if tech_count > disc_count:
            return "technical", 0.5
        elif disc_count > tech_count:
            return "discard", 0.5
        
        # Default to discard with low confidence if truly ambiguous
        return "discard", 0.3

    # =========================================================================
    # InferenceProvider interface (for compatibility)
    # =========================================================================

    async def generate(
        self, request: ChatCompletionRequest
    ) -> ChatCompletionResponse:
        """Generate completion (not used for vision - use classify_image)."""
        raise NotImplementedError(
            "DeepSeekVLProvider uses classify_image() for vision tasks, "
            "not generate(). Use LlamaCppProvider for text completions."
        )

    async def stream(
        self, request: ChatCompletionRequest
    ) -> AsyncIterator[ChatCompletionChunk]:
        """Stream completion (not supported for vision)."""
        raise NotImplementedError(
            "DeepSeekVLProvider does not support streaming. "
            "Use classify_image() for vision tasks."
        )
        # Yield statement to make this a generator (required for type)
        yield  # type: ignore[misc]

    def tokenize(self, text: str) -> list[int]:
        """Tokenize text."""
        if self._tokenizer is None:
            raise DeepSeekVLInferenceError("Model not loaded")
        return self._tokenizer.encode(text)

    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        return len(self.tokenize(text))
