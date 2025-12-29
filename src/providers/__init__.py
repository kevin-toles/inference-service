"""LLM inference providers for inference-service.

Providers:
- base: InferenceProvider ABC
- llamacpp: LlamaCppProvider (llama-cpp-python + Metal)
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


__all__: list[str] = [
    "InferenceProvider",
    "LlamaCppInferenceError",
    "LlamaCppModelLoadError",
    "LlamaCppModelNotFoundError",
    "LlamaCppProvider",
    "LlamaCppProviderError",
    "ModelMetadata",
]
