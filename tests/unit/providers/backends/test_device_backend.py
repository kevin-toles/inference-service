"""
Unit tests for DeviceBackend ABC.

Tests verify:
- ABC cannot be instantiated directly (AC-VLM2.1)
- Abstract methods are properly defined (AC-VLM2.2, AC-VLM2.3, AC-VLM2.4)

References:
    - VLM_DTYPE_COMPATIBILITY_WBS.md (WBS-VLM2)
    - GoF Design Patterns, Ch.5 Strategy Pattern
"""

import pytest
import torch
import torch.nn as nn

from src.providers.backends.device_backend import DeviceBackend


class TestDeviceBackendABC:
    """Tests for DeviceBackend abstract base class."""

    def test_abc_not_instantiable(self) -> None:
        """AC-VLM2.1: DeviceBackend ABC cannot be instantiated directly."""
        with pytest.raises(TypeError) as exc_info:
            DeviceBackend()  # type: ignore[abstract]
        
        assert "abstract" in str(exc_info.value).lower()

    def test_ensure_dtype_compatibility_is_abstract(self) -> None:
        """AC-VLM2.2: ensure_dtype_compatibility() is abstract method."""
        # Verify method exists and is abstract
        assert hasattr(DeviceBackend, "ensure_dtype_compatibility")
        assert getattr(
            DeviceBackend.ensure_dtype_compatibility, "__isabstractmethod__", False
        )

    def test_supports_dtype_is_abstract(self) -> None:
        """AC-VLM2.3: supports_dtype() is abstract method."""
        assert hasattr(DeviceBackend, "supports_dtype")
        assert getattr(
            DeviceBackend.supports_dtype, "__isabstractmethod__", False
        )

    def test_get_optimal_dtype_is_abstract(self) -> None:
        """AC-VLM2.4: get_optimal_dtype() is abstract method."""
        assert hasattr(DeviceBackend, "get_optimal_dtype")
        assert getattr(
            DeviceBackend.get_optimal_dtype, "__isabstractmethod__", False
        )


class TestConcreteBackendImplementation:
    """Tests that concrete implementations can be created."""

    def test_concrete_implementation_instantiable(self) -> None:
        """Concrete subclass implementing all methods can be instantiated."""
        
        class MockBackend(DeviceBackend):
            def ensure_dtype_compatibility(self, model: nn.Module) -> nn.Module:
                return model
            
            def supports_dtype(self, dtype: torch.dtype) -> bool:
                return True
            
            def get_optimal_dtype(self) -> torch.dtype:
                return torch.float32
        
        backend = MockBackend()
        assert isinstance(backend, DeviceBackend)

    def test_partial_implementation_not_instantiable(self) -> None:
        """Subclass missing abstract methods cannot be instantiated."""
        
        class IncompleteBackend(DeviceBackend):
            def ensure_dtype_compatibility(self, model: nn.Module) -> nn.Module:
                return model
            # Missing: supports_dtype, get_optimal_dtype
        
        with pytest.raises(TypeError) as exc_info:
            IncompleteBackend()  # type: ignore[abstract]
        
        assert "abstract" in str(exc_info.value).lower()
