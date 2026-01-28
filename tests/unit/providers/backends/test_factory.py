"""Tests for DeviceBackendFactory.

TDD RED Phase - WBS-VLM4: Implement DeviceBackendFactory

Acceptance Criteria:
- AC-VLM4.1: create_backend("mps") returns MPSBackend
- AC-VLM4.2: create_backend("mps:0") normalizes to MPSBackend
- AC-VLM4.3: create_backend("unknown") raises ValueError
- AC-VLM4.4: register_backend() allows custom backends
"""

import pytest

from src.providers.backends import DeviceBackend, MPSBackend
from src.providers.backends.factory import (
    DeviceBackendFactory,
    DeviceBackendFactoryError,
    DEVICE_MPS,
    DEVICE_CUDA,
    DEVICE_CPU,
)


class TestDeviceBackendFactory:
    """Tests for DeviceBackendFactory."""

    # =========================================================================
    # AC-VLM4.1: create_backend("mps") returns MPSBackend
    # =========================================================================

    def test_create_mps_backend(self) -> None:
        """create_backend('mps') returns MPSBackend instance."""
        backend = DeviceBackendFactory.create_backend("mps")
        assert isinstance(backend, MPSBackend)

    def test_create_mps_backend_uppercase(self) -> None:
        """create_backend('MPS') normalizes case and returns MPSBackend."""
        backend = DeviceBackendFactory.create_backend("MPS")
        assert isinstance(backend, MPSBackend)

    def test_create_mps_backend_mixed_case(self) -> None:
        """create_backend('Mps') normalizes case and returns MPSBackend."""
        backend = DeviceBackendFactory.create_backend("Mps")
        assert isinstance(backend, MPSBackend)

    # =========================================================================
    # AC-VLM4.2: create_backend("mps:0") normalizes to MPSBackend
    # =========================================================================

    def test_mps_device_index_zero(self) -> None:
        """create_backend('mps:0') normalizes to MPSBackend."""
        backend = DeviceBackendFactory.create_backend("mps:0")
        assert isinstance(backend, MPSBackend)

    def test_mps_device_index_one(self) -> None:
        """create_backend('mps:1') normalizes to MPSBackend."""
        backend = DeviceBackendFactory.create_backend("mps:1")
        assert isinstance(backend, MPSBackend)

    def test_cuda_device_index(self) -> None:
        """create_backend('cuda:0') normalizes device type."""
        # Note: CUDA backend not implemented yet, should raise
        with pytest.raises(DeviceBackendFactoryError):
            DeviceBackendFactory.create_backend("cuda:0")

    # =========================================================================
    # AC-VLM4.3: create_backend("unknown") raises ValueError
    # =========================================================================

    def test_unknown_device_raises_error(self) -> None:
        """create_backend('unknown') raises DeviceBackendFactoryError."""
        with pytest.raises(DeviceBackendFactoryError) as exc_info:
            DeviceBackendFactory.create_backend("unknown")
        assert "unknown" in str(exc_info.value).lower()

    def test_empty_device_raises_error(self) -> None:
        """create_backend('') raises DeviceBackendFactoryError."""
        with pytest.raises(DeviceBackendFactoryError):
            DeviceBackendFactory.create_backend("")

    def test_none_device_raises_error(self) -> None:
        """create_backend(None) raises appropriate error."""
        with pytest.raises((DeviceBackendFactoryError, TypeError, AttributeError)):
            DeviceBackendFactory.create_backend(None)  # type: ignore

    # =========================================================================
    # AC-VLM4.4: register_backend() allows custom backends
    # =========================================================================

    def test_register_custom_backend(self) -> None:
        """register_backend() allows registering custom backends."""
        # Create a custom backend
        class CustomBackend(DeviceBackend):
            def ensure_dtype_compatibility(self, model):
                return model

            def supports_dtype(self, dtype):
                return True

            def get_optimal_dtype(self):
                import torch
                return torch.float32

        # Register it
        DeviceBackendFactory.register_backend("custom", CustomBackend)

        # Create instance
        backend = DeviceBackendFactory.create_backend("custom")
        assert isinstance(backend, CustomBackend)

        # Cleanup: remove from registry
        DeviceBackendFactory._registry.pop("custom", None)

    def test_register_backend_overwrites_existing(self) -> None:
        """register_backend() can overwrite existing backends."""
        # Create a replacement MPS backend
        class ReplacementMPSBackend(DeviceBackend):
            def ensure_dtype_compatibility(self, model):
                return model

            def supports_dtype(self, dtype):
                return True

            def get_optimal_dtype(self):
                import torch
                return torch.float16

        # Save original
        original = DeviceBackendFactory._registry.get("mps")

        # Register replacement
        DeviceBackendFactory.register_backend("mps", ReplacementMPSBackend)

        # Verify it's replaced
        backend = DeviceBackendFactory.create_backend("mps")
        assert isinstance(backend, ReplacementMPSBackend)

        # Restore original
        if original:
            DeviceBackendFactory._registry["mps"] = original

    def test_register_backend_with_device_index(self) -> None:
        """register_backend() uses normalized device name."""
        class TestBackend(DeviceBackend):
            def ensure_dtype_compatibility(self, model):
                return model

            def supports_dtype(self, dtype):
                return True

            def get_optimal_dtype(self):
                import torch
                return torch.float32

        # Register with colon
        DeviceBackendFactory.register_backend("test:0", TestBackend)

        # Should be registered as "test:0" (exact key)
        assert "test:0" in DeviceBackendFactory._registry

        # Cleanup
        DeviceBackendFactory._registry.pop("test:0", None)


class TestDeviceConstants:
    """Tests for device name constants."""

    def test_device_mps_constant(self) -> None:
        """DEVICE_MPS constant equals 'mps'."""
        assert DEVICE_MPS == "mps"

    def test_device_cuda_constant(self) -> None:
        """DEVICE_CUDA constant equals 'cuda'."""
        assert DEVICE_CUDA == "cuda"

    def test_device_cpu_constant(self) -> None:
        """DEVICE_CPU constant equals 'cpu'."""
        assert DEVICE_CPU == "cpu"


class TestDeviceBackendFactoryError:
    """Tests for DeviceBackendFactoryError exception."""

    def test_exception_is_value_error_subclass(self) -> None:
        """DeviceBackendFactoryError is a ValueError subclass."""
        assert issubclass(DeviceBackendFactoryError, ValueError)

    def test_exception_message(self) -> None:
        """DeviceBackendFactoryError preserves message."""
        error = DeviceBackendFactoryError("test message")
        assert str(error) == "test message"
