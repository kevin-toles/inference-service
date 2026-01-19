# Hardware Abstraction Layer Specification

**Version**: 1.0.0  
**Created**: January 18, 2026  
**Status**: DESIGN (Not Implemented)  
**WBS Reference**: WBS_B5_B1_REMAINING_WORK.md - GPU-5

---

## Overview

This document specifies the `HardwareAllocator` interface for abstracting GPU and compute resource allocation in the inference-service.

**Purpose:**
- Decouple inference code from hardware-specific APIs
- Support multiple backends (Metal, CUDA, ROCm, CPU)
- Enable future multi-GPU and distributed inference
- Provide testable abstraction for unit tests

**Status:** This is a **design specification only**. Implementation is planned but not yet scheduled.

---

## 1. Interface Design

### Core Abstraction

```python
# src/hardware/allocator.py (PROPOSED)

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Optional, List


class DeviceType(Enum):
    """Supported compute device types."""
    CPU = "cpu"
    CUDA = "cuda"        # NVIDIA
    METAL = "metal"      # Apple Silicon
    ROCM = "rocm"        # AMD
    VULKAN = "vulkan"    # Cross-platform (future)


@dataclass(frozen=True)
class DeviceInfo:
    """Information about a compute device.
    
    Attributes:
        device_id: Unique device identifier (e.g., "cuda:0", "metal:0")
        device_type: Type of compute device
        name: Human-readable device name (e.g., "NVIDIA RTX 4090")
        memory_total_gb: Total device memory in gigabytes
        memory_available_gb: Currently available memory in gigabytes
        compute_capability: Device compute version (e.g., "8.9" for Ada)
        is_available: Whether device is currently usable
    """
    device_id: str
    device_type: DeviceType
    name: str
    memory_total_gb: float
    memory_available_gb: float
    compute_capability: Optional[str] = None
    is_available: bool = True


@dataclass
class AllocationRequest:
    """Request to allocate compute resources.
    
    Attributes:
        memory_required_gb: Minimum memory needed
        preferred_device_type: Preferred device type (None = any)
        preferred_device_id: Specific device to use (None = auto-select)
        allow_fallback: Whether to fallback to other devices if preferred unavailable
    """
    memory_required_gb: float
    preferred_device_type: Optional[DeviceType] = None
    preferred_device_id: Optional[str] = None
    allow_fallback: bool = True


@dataclass
class AllocationResult:
    """Result of a compute resource allocation.
    
    Attributes:
        success: Whether allocation succeeded
        device: Allocated device info (None if failed)
        allocation_id: Unique ID for this allocation (for release)
        error_message: Error message if allocation failed
    """
    success: bool
    device: Optional[DeviceInfo] = None
    allocation_id: Optional[str] = None
    error_message: Optional[str] = None


class HardwareAllocator(ABC):
    """Abstract interface for hardware resource allocation.
    
    This interface abstracts GPU/CPU allocation to enable:
    - Multiple backend support (Metal, CUDA, ROCm)
    - Unit testing with mock allocators
    - Future multi-GPU and distributed inference
    
    Reference: ARCHITECTURE_ROUNDTABLE_FINDINGS.md R6
    """
    
    @abstractmethod
    def discover_devices(self) -> List[DeviceInfo]:
        """Discover available compute devices.
        
        Returns:
            List of available devices with their capabilities.
        """
        pass
    
    @abstractmethod
    def allocate(self, request: AllocationRequest) -> AllocationResult:
        """Allocate compute resources for inference.
        
        Args:
            request: Allocation requirements and preferences.
            
        Returns:
            AllocationResult indicating success/failure and allocated device.
        """
        pass
    
    @abstractmethod
    def release(self, allocation_id: str) -> bool:
        """Release a previous allocation.
        
        Args:
            allocation_id: ID from AllocationResult.
            
        Returns:
            True if released successfully.
        """
        pass
    
    @abstractmethod
    def get_device_status(self, device_id: str) -> Optional[DeviceInfo]:
        """Get current status of a specific device.
        
        Args:
            device_id: Device identifier.
            
        Returns:
            DeviceInfo or None if device not found.
        """
        pass
    
    @property
    @abstractmethod
    def default_device_type(self) -> DeviceType:
        """Get the default device type for this platform."""
        pass
```

---

## 2. Backend Implementations

### MetalAllocator (macOS)

```python
# src/hardware/metal_allocator.py (PROPOSED)

import platform
from typing import List, Optional

from src.hardware.allocator import (
    HardwareAllocator,
    DeviceType,
    DeviceInfo,
    AllocationRequest,
    AllocationResult,
)


class MetalAllocator(HardwareAllocator):
    """Hardware allocator for Apple Metal (macOS).
    
    Apple Silicon uses unified memory architecture where GPU
    and CPU share the same memory pool.
    
    Key differences from discrete GPUs:
    - No separate GPU memory to manage
    - Single "device" representing the unified GPU
    - Memory limits based on system RAM
    """
    
    def __init__(self) -> None:
        self._allocations: dict[str, DeviceInfo] = {}
        self._allocation_counter = 0
    
    def discover_devices(self) -> List[DeviceInfo]:
        """Discover Metal devices (single unified GPU on Apple Silicon)."""
        if platform.system() != "Darwin":
            return []
        
        # Query system for GPU info
        # In real implementation, use Metal Python bindings or system_profiler
        return [
            DeviceInfo(
                device_id="metal:0",
                device_type=DeviceType.METAL,
                name="Apple M-series GPU",
                memory_total_gb=self._get_system_memory_gb(),
                memory_available_gb=self._get_available_memory_gb(),
                compute_capability=None,  # Not applicable for Metal
                is_available=True,
            )
        ]
    
    def allocate(self, request: AllocationRequest) -> AllocationResult:
        """Allocate Metal compute resources."""
        devices = self.discover_devices()
        if not devices:
            return AllocationResult(
                success=False,
                error_message="No Metal devices available"
            )
        
        device = devices[0]  # Single unified GPU
        
        if device.memory_available_gb < request.memory_required_gb:
            return AllocationResult(
                success=False,
                error_message=f"Insufficient memory: need {request.memory_required_gb}GB, "
                              f"have {device.memory_available_gb}GB"
            )
        
        self._allocation_counter += 1
        allocation_id = f"metal-alloc-{self._allocation_counter}"
        self._allocations[allocation_id] = device
        
        return AllocationResult(
            success=True,
            device=device,
            allocation_id=allocation_id,
        )
    
    def release(self, allocation_id: str) -> bool:
        """Release Metal allocation."""
        if allocation_id in self._allocations:
            del self._allocations[allocation_id]
            return True
        return False
    
    def get_device_status(self, device_id: str) -> Optional[DeviceInfo]:
        """Get Metal device status."""
        devices = self.discover_devices()
        for device in devices:
            if device.device_id == device_id:
                return device
        return None
    
    @property
    def default_device_type(self) -> DeviceType:
        return DeviceType.METAL
    
    def _get_system_memory_gb(self) -> float:
        """Get total system memory in GB."""
        import subprocess
        result = subprocess.run(
            ["sysctl", "-n", "hw.memsize"],
            capture_output=True,
            text=True
        )
        return int(result.stdout.strip()) / (1024**3)
    
    def _get_available_memory_gb(self) -> float:
        """Get available system memory in GB."""
        # Simplified - real implementation would use vm_stat
        return self._get_system_memory_gb() * 0.8  # Assume 80% available
```

### CUDAAllocator (Linux/Windows)

```python
# src/hardware/cuda_allocator.py (PROPOSED)

import os
from typing import List, Optional

from src.hardware.allocator import (
    HardwareAllocator,
    DeviceType,
    DeviceInfo,
    AllocationRequest,
    AllocationResult,
)


class CUDAAllocator(HardwareAllocator):
    """Hardware allocator for NVIDIA CUDA GPUs.
    
    Supports multi-GPU systems with explicit device selection
    via CUDA_VISIBLE_DEVICES or device index.
    """
    
    def __init__(self) -> None:
        self._allocations: dict[str, DeviceInfo] = {}
        self._allocation_counter = 0
    
    def discover_devices(self) -> List[DeviceInfo]:
        """Discover CUDA devices using pynvml or nvidia-smi."""
        try:
            import pynvml
            pynvml.nvmlInit()
            device_count = pynvml.nvmlDeviceGetCount()
            
            devices = []
            for i in range(device_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                name = pynvml.nvmlDeviceGetName(handle)
                memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
                
                devices.append(DeviceInfo(
                    device_id=f"cuda:{i}",
                    device_type=DeviceType.CUDA,
                    name=name.decode() if isinstance(name, bytes) else name,
                    memory_total_gb=memory.total / (1024**3),
                    memory_available_gb=memory.free / (1024**3),
                    compute_capability=self._get_compute_capability(handle),
                    is_available=True,
                ))
            
            pynvml.nvmlShutdown()
            return devices
            
        except ImportError:
            return []  # pynvml not installed
        except Exception:
            return []  # No CUDA devices
    
    def allocate(self, request: AllocationRequest) -> AllocationResult:
        """Allocate CUDA compute resources."""
        devices = self.discover_devices()
        if not devices:
            return AllocationResult(
                success=False,
                error_message="No CUDA devices available"
            )
        
        # Select device
        target_device = None
        if request.preferred_device_id:
            for device in devices:
                if device.device_id == request.preferred_device_id:
                    target_device = device
                    break
        else:
            # Auto-select: find device with enough memory
            for device in devices:
                if device.memory_available_gb >= request.memory_required_gb:
                    target_device = device
                    break
        
        if not target_device:
            if request.allow_fallback:
                # Try any device with enough memory
                for device in devices:
                    if device.memory_available_gb >= request.memory_required_gb:
                        target_device = device
                        break
        
        if not target_device:
            return AllocationResult(
                success=False,
                error_message=f"No device with {request.memory_required_gb}GB available"
            )
        
        # Set CUDA_VISIBLE_DEVICES for this allocation
        device_index = int(target_device.device_id.split(":")[1])
        os.environ["CUDA_VISIBLE_DEVICES"] = str(device_index)
        
        self._allocation_counter += 1
        allocation_id = f"cuda-alloc-{self._allocation_counter}"
        self._allocations[allocation_id] = target_device
        
        return AllocationResult(
            success=True,
            device=target_device,
            allocation_id=allocation_id,
        )
    
    def release(self, allocation_id: str) -> bool:
        """Release CUDA allocation."""
        if allocation_id in self._allocations:
            del self._allocations[allocation_id]
            return True
        return False
    
    def get_device_status(self, device_id: str) -> Optional[DeviceInfo]:
        """Get CUDA device status."""
        devices = self.discover_devices()
        for device in devices:
            if device.device_id == device_id:
                return device
        return None
    
    @property
    def default_device_type(self) -> DeviceType:
        return DeviceType.CUDA
    
    def _get_compute_capability(self, handle) -> str:
        """Get CUDA compute capability."""
        import pynvml
        major = pynvml.nvmlDeviceGetCudaComputeCapability(handle)[0]
        minor = pynvml.nvmlDeviceGetCudaComputeCapability(handle)[1]
        return f"{major}.{minor}"
```

---

## 3. Factory Pattern

```python
# src/hardware/factory.py (PROPOSED)

import platform
from typing import Optional

from src.hardware.allocator import HardwareAllocator, DeviceType
from src.hardware.metal_allocator import MetalAllocator
from src.hardware.cuda_allocator import CUDAAllocator


class HardwareAllocatorFactory:
    """Factory for creating platform-appropriate HardwareAllocator.
    
    Automatically selects the correct allocator based on:
    1. Explicit device_type preference
    2. Platform detection (macOS → Metal, Linux/Windows → CUDA)
    3. Fallback to CPU-only allocator
    """
    
    @staticmethod
    def create(
        device_type: Optional[DeviceType] = None
    ) -> HardwareAllocator:
        """Create a HardwareAllocator for the current platform.
        
        Args:
            device_type: Explicit device type preference.
                         None = auto-detect based on platform.
        
        Returns:
            Appropriate HardwareAllocator implementation.
        """
        if device_type == DeviceType.METAL:
            return MetalAllocator()
        
        if device_type == DeviceType.CUDA:
            return CUDAAllocator()
        
        # Auto-detect
        if platform.system() == "Darwin":
            return MetalAllocator()
        
        # Try CUDA first (most common)
        cuda_allocator = CUDAAllocator()
        if cuda_allocator.discover_devices():
            return cuda_allocator
        
        # Fallback to CPU-only
        return CPUAllocator()


class CPUAllocator(HardwareAllocator):
    """Fallback allocator for CPU-only inference."""
    
    def discover_devices(self) -> list:
        import psutil
        return [DeviceInfo(
            device_id="cpu:0",
            device_type=DeviceType.CPU,
            name="CPU",
            memory_total_gb=psutil.virtual_memory().total / (1024**3),
            memory_available_gb=psutil.virtual_memory().available / (1024**3),
            is_available=True,
        )]
    
    def allocate(self, request):
        devices = self.discover_devices()
        device = devices[0]
        return AllocationResult(
            success=True,
            device=device,
            allocation_id="cpu-alloc-0",
        )
    
    def release(self, allocation_id: str) -> bool:
        return True
    
    def get_device_status(self, device_id: str):
        return self.discover_devices()[0]
    
    @property
    def default_device_type(self) -> DeviceType:
        return DeviceType.CPU
```

---

## 4. Integration with LlamaCppProvider

### Proposed Changes

```python
# src/providers/llamacpp.py (PROPOSED CHANGES)

from src.hardware.factory import HardwareAllocatorFactory
from src.hardware.allocator import AllocationRequest

class LlamaCppProvider(InferenceProvider):
    
    def __init__(
        self,
        model_path: Path,
        model_id: str,
        context_length: int = DEFAULT_CONTEXT_LENGTH,
        n_gpu_layers: int = 0,
        memory_mb: int = 0,
        allocator: Optional[HardwareAllocator] = None,  # NEW
    ) -> None:
        # ... existing init ...
        
        # Use provided allocator or create default
        self._allocator = allocator or HardwareAllocatorFactory.create()
        self._allocation_id: Optional[str] = None
    
    async def load(self) -> None:
        """Load model with hardware allocation."""
        if self._is_loaded:
            return
        
        # Request hardware allocation
        request = AllocationRequest(
            memory_required_gb=self._memory_mb / 1024,
            allow_fallback=True,
        )
        result = self._allocator.allocate(request)
        
        if not result.success:
            raise LlamaCppModelLoadError(
                f"Hardware allocation failed: {result.error_message}"
            )
        
        self._allocation_id = result.allocation_id
        
        # Configure GPU layers based on allocated device
        n_gpu_layers = self._n_gpu_layers
        if result.device.device_type == DeviceType.CPU:
            n_gpu_layers = 0  # Force CPU-only
        
        # ... rest of load() ...
    
    async def unload(self) -> None:
        """Unload model and release hardware allocation."""
        if self._model is not None:
            self._model = None
        self._is_loaded = False
        
        # Release hardware allocation
        if self._allocation_id:
            self._allocator.release(self._allocation_id)
            self._allocation_id = None
```

---

## 5. Testing Strategy

### Mock Allocator for Unit Tests

```python
# tests/mocks/mock_allocator.py (PROPOSED)

from src.hardware.allocator import (
    HardwareAllocator,
    DeviceType,
    DeviceInfo,
    AllocationRequest,
    AllocationResult,
)


class MockHardwareAllocator(HardwareAllocator):
    """Mock allocator for unit testing.
    
    Allows tests to control allocation behavior without
    requiring actual GPU hardware.
    """
    
    def __init__(
        self,
        devices: list[DeviceInfo] = None,
        should_fail: bool = False,
        fail_message: str = "Mock allocation failure",
    ) -> None:
        self.devices = devices or [
            DeviceInfo(
                device_id="mock:0",
                device_type=DeviceType.METAL,
                name="Mock GPU",
                memory_total_gb=16.0,
                memory_available_gb=12.0,
            )
        ]
        self.should_fail = should_fail
        self.fail_message = fail_message
        self.allocations: list[str] = []
    
    def discover_devices(self) -> list[DeviceInfo]:
        return self.devices
    
    def allocate(self, request: AllocationRequest) -> AllocationResult:
        if self.should_fail:
            return AllocationResult(
                success=False,
                error_message=self.fail_message,
            )
        
        allocation_id = f"mock-alloc-{len(self.allocations)}"
        self.allocations.append(allocation_id)
        
        return AllocationResult(
            success=True,
            device=self.devices[0],
            allocation_id=allocation_id,
        )
    
    def release(self, allocation_id: str) -> bool:
        if allocation_id in self.allocations:
            self.allocations.remove(allocation_id)
            return True
        return False
    
    def get_device_status(self, device_id: str) -> DeviceInfo | None:
        for device in self.devices:
            if device.device_id == device_id:
                return device
        return None
    
    @property
    def default_device_type(self) -> DeviceType:
        return DeviceType.METAL
```

---

## 6. Implementation Roadmap

| Phase | Tasks | Effort |
|-------|-------|--------|
| **Phase 1** | Interface definition, CPUAllocator | 2 days |
| **Phase 2** | MetalAllocator, basic tests | 2 days |
| **Phase 3** | CUDAAllocator, multi-GPU tests | 3 days |
| **Phase 4** | LlamaCppProvider integration | 2 days |
| **Phase 5** | ROCmAllocator (optional) | 2 days |

**Total Estimated Effort:** 9-11 days

---

## 7. Design Decisions

### Why an Interface?

1. **Testability** - Mock allocators for unit tests without GPU
2. **Portability** - Same code works on macOS, Linux, Windows
3. **Extensibility** - Easy to add ROCm, Vulkan, TPU support
4. **Separation of Concerns** - Inference logic separate from hardware management

### Why Not Use llama-cpp-python Directly?

llama-cpp-python handles GPU selection internally, but:
- No visibility into available memory before loading
- No way to reserve GPU resources across multiple models
- No cross-platform abstraction for device discovery

The HardwareAllocator provides a layer that can:
- Query available resources before loading
- Coordinate GPU usage across multiple model instances
- Provide consistent API across different backends

---

## Citations

| # | Source | Section |
|---|--------|---------|
| [^1] | ARCHITECTURE_ROUNDTABLE_FINDINGS.md | §R6: Hardware Abstraction Interface |
| [^2] | GPU_ALLOCATION.md | §4: Multi-GPU Selection Rules |
| [^3] | Python Cookbook 3rd Ed | Ch.9: Abstract Base Classes |
| [^4] | Design Patterns (GoF) | Factory Pattern |

