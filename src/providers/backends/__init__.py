"""
Device-specific backend implementations for dtype compatibility.

This module provides the DeviceBackend pattern for handling dtype mismatches
across different compute backends (MPS, CUDA, CPU). Primary use case is
converting bfloat16 weights to float32 on Apple Silicon MPS.

Exports:
    DeviceBackend: Abstract base class for device backends
    MPSBackend: MPS-specific dtype handling (bfloat16 → float32)
    CPUBackend: CPU-specific dtype handling (bfloat16 → float32)
    DeviceBackendFactory: Factory for creating backends by device string

Example:
    >>> from src.providers.backends import DeviceBackendFactory
    >>> backend = DeviceBackendFactory.create_backend("mps")
    >>> model = backend.ensure_dtype_compatibility(model)

References:
    - DTYPE_COMPATIBILITY_ARCHITECTURE.md
"""

from .device_backend import CPUBackend, DeviceBackend, MPSBackend
from .factory import (
    DeviceBackendFactory,
    DeviceBackendFactoryError,
    DEVICE_MPS,
    DEVICE_CUDA,
    DEVICE_CPU,
)

__all__ = [
    "DeviceBackend",
    "MPSBackend",
    "CPUBackend",
    "DeviceBackendFactory",
    "DeviceBackendFactoryError",
    "DEVICE_MPS",
    "DEVICE_CUDA",
    "DEVICE_CPU",
]
