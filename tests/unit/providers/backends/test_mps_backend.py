"""
Unit tests for MPSBackend.

Tests verify:
- bfloat16 parameters converted to float32 (AC-VLM3.1)
- Parameter semantics preserved (AC-VLM3.2)
- Buffers converted via _buffers dict (AC-VLM3.3)
- Shared/tied weights converted only once (AC-VLM3.4)
- supports_dtype(bfloat16) returns False (AC-VLM3.5)
- get_optimal_dtype() returns float32 (AC-VLM3.6)

References:
    - VLM_DTYPE_COMPATIBILITY_WBS.md (WBS-VLM3)
    - GPT-4o Multi-LLM Consensus [^5]
    - PyTorch torch.nn.Parameter documentation [^1]
"""

import pytest
import torch
import torch.nn as nn

from src.providers.backends.device_backend import DeviceBackend, MPSBackend


class TestMPSBackendParameterConversion:
    """Tests for bfloat16 → float32 parameter conversion (AC-VLM3.1)."""

    def test_bfloat16_to_float32(self) -> None:
        """AC-VLM3.1: bfloat16 parameters are converted to float32."""
        # Create a simple model with bfloat16 weights
        model = nn.Linear(10, 5)
        model.weight.data = model.weight.data.to(torch.bfloat16)
        model.bias.data = model.bias.data.to(torch.bfloat16)
        
        assert model.weight.dtype == torch.bfloat16
        assert model.bias.dtype == torch.bfloat16
        
        backend = MPSBackend()
        model = backend.ensure_dtype_compatibility(model)
        
        assert model.weight.dtype == torch.float32
        assert model.bias.dtype == torch.float32

    def test_float32_unchanged(self) -> None:
        """float32 parameters should remain unchanged."""
        model = nn.Linear(10, 5)  # Default is float32
        original_weight = model.weight.data.clone()
        
        backend = MPSBackend()
        model = backend.ensure_dtype_compatibility(model)
        
        assert model.weight.dtype == torch.float32
        assert torch.allclose(model.weight.data, original_weight)

    def test_float16_unchanged(self) -> None:
        """float16 parameters should remain unchanged (MPS supports float16)."""
        model = nn.Linear(10, 5)
        model.weight.data = model.weight.data.to(torch.float16)
        
        backend = MPSBackend()
        model = backend.ensure_dtype_compatibility(model)
        
        assert model.weight.dtype == torch.float16

    def test_nested_modules_converted(self) -> None:
        """Nested module parameters should all be converted."""
        model = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 5),
        )
        # Convert all to bfloat16
        for param in model.parameters():
            param.data = param.data.to(torch.bfloat16)
        
        backend = MPSBackend()
        model = backend.ensure_dtype_compatibility(model)
        
        for param in model.parameters():
            assert param.dtype == torch.float32, f"Parameter still {param.dtype}"


class TestMPSBackendParameterSemantics:
    """Tests for Parameter semantics preservation (AC-VLM3.2)."""

    def test_parameter_not_tensor(self) -> None:
        """AC-VLM3.2: Parameters remain Parameters (not converted to Tensors)."""
        model = nn.Linear(10, 5)
        model.weight.data = model.weight.data.to(torch.bfloat16)
        
        # Verify it's a Parameter before conversion
        assert isinstance(model.weight, nn.Parameter)
        
        backend = MPSBackend()
        model = backend.ensure_dtype_compatibility(model)
        
        # CRITICAL: Must still be a Parameter, not a Tensor
        assert isinstance(model.weight, nn.Parameter), \
            "Weight was converted to Tensor, breaking gradient tracking!"

    def test_requires_grad_preserved(self) -> None:
        """requires_grad flag should be preserved after conversion."""
        model = nn.Linear(10, 5)
        model.weight.data = model.weight.data.to(torch.bfloat16)
        model.weight.requires_grad = True
        
        backend = MPSBackend()
        model = backend.ensure_dtype_compatibility(model)
        
        assert model.weight.requires_grad is True

    def test_requires_grad_false_preserved(self) -> None:
        """requires_grad=False should be preserved."""
        model = nn.Linear(10, 5)
        model.weight.data = model.weight.data.to(torch.bfloat16)
        model.weight.requires_grad = False
        
        backend = MPSBackend()
        model = backend.ensure_dtype_compatibility(model)
        
        assert model.weight.requires_grad is False


class TestMPSBackendBufferConversion:
    """Tests for buffer conversion (AC-VLM3.3)."""

    def test_buffers_converted(self) -> None:
        """AC-VLM3.3: Buffers are converted via _buffers dict."""
        class ModelWithBuffer(nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("running_mean", torch.zeros(10, dtype=torch.bfloat16))
                self.register_buffer("running_var", torch.ones(10, dtype=torch.bfloat16))
        
        model = ModelWithBuffer()
        assert model.running_mean.dtype == torch.bfloat16
        assert model.running_var.dtype == torch.bfloat16
        
        backend = MPSBackend()
        model = backend.ensure_dtype_compatibility(model)
        
        assert model.running_mean.dtype == torch.float32
        assert model.running_var.dtype == torch.float32

    def test_none_buffer_handled(self) -> None:
        """None buffers should be handled gracefully."""
        class ModelWithNoneBuffer(nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("optional_buffer", None)
        
        model = ModelWithNoneBuffer()
        
        backend = MPSBackend()
        # Should not raise
        model = backend.ensure_dtype_compatibility(model)
        
        assert model.optional_buffer is None


class TestMPSBackendSharedWeights:
    """Tests for shared/tied weights handling (AC-VLM3.4)."""

    def test_shared_weights_once(self) -> None:
        """AC-VLM3.4: Shared/tied weights should be converted only once."""
        class ModelWithTiedWeights(nn.Module):
            def __init__(self):
                super().__init__()
                self.embed = nn.Embedding(100, 64)
                self.output = nn.Linear(64, 100, bias=False)
                # Tie weights
                self.output.weight = self.embed.weight
        
        model = ModelWithTiedWeights()
        model.embed.weight.data = model.embed.weight.data.to(torch.bfloat16)
        
        # Verify weights are actually tied (same object)
        assert model.embed.weight is model.output.weight
        
        backend = MPSBackend()
        model = backend.ensure_dtype_compatibility(model)
        
        # Both should be converted
        assert model.embed.weight.dtype == torch.float32
        assert model.output.weight.dtype == torch.float32
        
        # And still tied
        assert model.embed.weight is model.output.weight


class TestMPSBackendDtypeSupport:
    """Tests for dtype support methods (AC-VLM3.5, AC-VLM3.6)."""

    def test_supports_bfloat16_false(self) -> None:
        """AC-VLM3.5: supports_dtype(bfloat16) returns False."""
        backend = MPSBackend()
        assert backend.supports_dtype(torch.bfloat16) is False

    def test_supports_float32_true(self) -> None:
        """supports_dtype(float32) returns True."""
        backend = MPSBackend()
        assert backend.supports_dtype(torch.float32) is True

    def test_supports_float16_true(self) -> None:
        """supports_dtype(float16) returns True (MPS supports float16)."""
        backend = MPSBackend()
        assert backend.supports_dtype(torch.float16) is True

    def test_optimal_dtype_float32(self) -> None:
        """AC-VLM3.6: get_optimal_dtype() returns float32."""
        backend = MPSBackend()
        assert backend.get_optimal_dtype() == torch.float32


class TestMPSBackendIsDeviceBackend:
    """Tests that MPSBackend is a proper DeviceBackend implementation."""

    def test_is_device_backend(self) -> None:
        """MPSBackend is a DeviceBackend subclass."""
        backend = MPSBackend()
        assert isinstance(backend, DeviceBackend)

    def test_returns_same_model_instance(self) -> None:
        """ensure_dtype_compatibility returns the same model instance."""
        model = nn.Linear(10, 5)
        model.weight.data = model.weight.data.to(torch.bfloat16)
        
        backend = MPSBackend()
        result = backend.ensure_dtype_compatibility(model)
        
        assert result is model
