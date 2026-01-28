"""Device backend factory for creating device-specific backends.

Per GoF Factory pattern, this factory creates the appropriate backend
based on device string, allowing transparent backend selection.

References:
    - Gamma et al., Design Patterns: GoF, Ch.3, §Factory Method, pp.107-116
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .device_backend import CPUBackend, DeviceBackend, MPSBackend

if TYPE_CHECKING:
    pass

# =============================================================================
# Device Constants (AP-1: No duplicated string literals)
# =============================================================================
DEVICE_MPS = "mps"
DEVICE_CUDA = "cuda"
DEVICE_CPU = "cpu"


# =============================================================================
# Custom Exception (AP-5: Exceptions use {Service}Error prefix)
# =============================================================================
class DeviceBackendFactoryError(ValueError):
    """Raised when factory cannot create a backend for the given device."""

    pass


# =============================================================================
# Factory Implementation
# =============================================================================
class DeviceBackendFactory:
    """Factory for creating device-specific backends.

    This factory implements the GoF Factory Method pattern to create
    device-specific backends based on a device string. It supports:

    - Device type normalization (e.g., "MPS:0" -> "mps")
    - Custom backend registration via register_backend()
    - Clear error messages for unknown devices

    Example:
        >>> backend = DeviceBackendFactory.create_backend("mps")
        >>> isinstance(backend, MPSBackend)
        True

        >>> backend = DeviceBackendFactory.create_backend("mps:0")
        >>> isinstance(backend, MPSBackend)
        True
    """

    _registry: dict[str, type[DeviceBackend]] = {
        DEVICE_MPS: MPSBackend,
        DEVICE_CPU: CPUBackend,
    }

    @classmethod
    def create_backend(cls, device: str) -> DeviceBackend:
        """Create a backend instance for the given device.

        Args:
            device: Device string (e.g., "mps", "mps:0", "cuda:1").
                    Case-insensitive. Device indices are stripped.

        Returns:
            DeviceBackend instance appropriate for the device.

        Raises:
            DeviceBackendFactoryError: If device type is unknown or invalid.

        Example:
            >>> backend = DeviceBackendFactory.create_backend("mps:0")
            >>> backend.get_optimal_dtype()
            torch.float32
        """
        if not device:
            raise DeviceBackendFactoryError(
                "Device string cannot be empty. "
                f"Supported devices: {list(cls._registry.keys())}"
            )

        # Normalize device string: "MPS:0" -> "mps"
        device_type = device.split(":")[0].lower()

        if device_type not in cls._registry:
            raise DeviceBackendFactoryError(
                f"Unknown device type: '{device_type}'. "
                f"Supported devices: {list(cls._registry.keys())}. "
                "Use register_backend() to add custom backends."
            )

        return cls._registry[device_type]()

    @classmethod
    def register_backend(
        cls,
        device: str,
        backend_class: type[DeviceBackend],
    ) -> None:
        """Register a custom backend for a device type.

        This allows extending the factory with new device backends
        without modifying the factory itself (Open/Closed Principle).

        Args:
            device: Device name to register (e.g., "tpu", "npu").
                    Will be stored exactly as provided.
            backend_class: DeviceBackend subclass to instantiate.

        Example:
            >>> class TPUBackend(DeviceBackend):
            ...     def ensure_dtype_compatibility(self, model):
            ...         return model
            ...     def supports_dtype(self, dtype):
            ...         return True
            ...     def get_optimal_dtype(self):
            ...         return torch.bfloat16
            >>> DeviceBackendFactory.register_backend("tpu", TPUBackend)
            >>> backend = DeviceBackendFactory.create_backend("tpu")
        """
        cls._registry[device] = backend_class

    @classmethod
    def get_registered_devices(cls) -> list[str]:
        """Return list of registered device types.

        Returns:
            List of device names that have registered backends.
        """
        return list(cls._registry.keys())
