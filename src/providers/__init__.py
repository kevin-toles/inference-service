"""LLM inference providers for inference-service.

Providers:
- base: InferenceProvider ABC
- llamacpp: LlamaCppProvider (llama-cpp-python + Metal)
- deepseek_vl: DeepSeekVLProvider (Vision-Language Model)
- vllm: VLLMProvider (future, CUDA)
"""

from src.providers.base import InferenceProvider, ModelMetadata
from src.providers.llamacpp import (
    LlamaCppInferenceError,
    LlamaCppModelLoadError,
    LlamaCppModelNotFoundError,
    LlamaCppProvider,
    LlamaCppProviderError,
)
from src.providers.deepseek_vl import (
    DeepSeekVLProvider,
    DeepSeekVLProviderError,
    DeepSeekVLModelNotFoundError,
    DeepSeekVLModelLoadError,
    DeepSeekVLInferenceError,
    DeepSeekVLImageError,
    VisionClassifyRequest,
    VisionClassifyResponse,
)


__all__: list[str] = [
    # Base
    "InferenceProvider",
    "ModelMetadata",
    # LlamaCpp
    "LlamaCppInferenceError",
    "LlamaCppModelLoadError",
    "LlamaCppModelNotFoundError",
    "LlamaCppProvider",
    "LlamaCppProviderError",
    # DeepSeek VL
    "DeepSeekVLProvider",
    "DeepSeekVLProviderError",
    "DeepSeekVLModelNotFoundError",
    "DeepSeekVLModelLoadError",
    "DeepSeekVLInferenceError",
    "DeepSeekVLImageError",
    "VisionClassifyRequest",
    "VisionClassifyResponse",
]
