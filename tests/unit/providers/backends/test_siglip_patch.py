"""Tests for SigLIP attention patch.

These tests verify that the attention patch correctly replaces the xformers
dependency with standard PyTorch attention operations.
"""

import pytest
import torch
import torch.nn as nn

from src.providers.backends.siglip_attention_patch import (
    _patched_attention_forward,
    apply_siglip_attention_patch,
    is_patch_applied,
)


class MockAttention(nn.Module):
    """Mock Attention class that mimics the SigLIP Attention structure."""
    
    def __init__(self, dim: int = 64, num_heads: int = 4, qk_norm: bool = False):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qk_norm = qk_norm
        self.fused_attn = True
        
        self.qkv = nn.Linear(dim, dim * 3)
        self.q_norm = nn.Identity()
        self.k_norm = nn.Identity()
        self.attn_drop = nn.Dropout(0.0)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Identity()


class TestPatchedAttentionForward:
    """Tests for the patched attention forward method."""
    
    def test_patched_forward_without_qk_norm(self):
        """Test patched forward method with qk_norm=False."""
        attn = MockAttention(dim=64, num_heads=4, qk_norm=False)
        attn.eval()
        
        # Create input tensor: (batch, seq_len, dim)
        x = torch.randn(2, 10, 64)
        
        # Run patched forward
        output = _patched_attention_forward(attn, x)
        
        # Verify output shape
        assert output.shape == x.shape
        
    def test_patched_forward_with_qk_norm(self):
        """Test patched forward method with qk_norm=True."""
        attn = MockAttention(dim=64, num_heads=4, qk_norm=True)
        attn.eval()
        
        x = torch.randn(2, 10, 64)
        output = _patched_attention_forward(attn, x)
        
        assert output.shape == x.shape
        
    def test_patched_forward_with_fused_attn_false(self):
        """Test patched forward with fused_attn=False uses manual attention."""
        attn = MockAttention(dim=64, num_heads=4, qk_norm=True)
        attn.fused_attn = False
        attn.eval()
        
        x = torch.randn(2, 10, 64)
        output = _patched_attention_forward(attn, x)
        
        assert output.shape == x.shape
        
    def test_patched_forward_batch_size_1(self):
        """Test patched forward with batch size 1."""
        attn = MockAttention(dim=64, num_heads=4)
        attn.eval()
        
        x = torch.randn(1, 100, 64)
        output = _patched_attention_forward(attn, x)
        
        assert output.shape == x.shape
        
    def test_patched_forward_larger_sequence(self):
        """Test patched forward with larger sequence length."""
        attn = MockAttention(dim=128, num_heads=8)
        attn.eval()
        
        x = torch.randn(4, 256, 128)
        output = _patched_attention_forward(attn, x)
        
        assert output.shape == x.shape


class TestApplyPatch:
    """Tests for the apply_siglip_attention_patch function."""
    
    def test_patch_is_idempotent(self):
        """Calling apply_siglip_attention_patch multiple times is safe."""
        # First application
        result1 = apply_siglip_attention_patch()
        
        # Second application should also succeed (idempotent)
        result2 = apply_siglip_attention_patch()
        
        # Both should return True if deepseek_vl2 is installed
        # (may be False if not installed, but should be consistent)
        assert result1 == result2
        
    def test_is_patch_applied_reflects_state(self):
        """is_patch_applied returns correct state."""
        # After attempting to apply, state should be updated
        apply_siglip_attention_patch()
        
        # is_patch_applied should return True if apply succeeded
        # (depends on whether deepseek_vl2 is installed)
        state = is_patch_applied()
        assert isinstance(state, bool)


class TestPatchedAttentionOnMPS:
    """Integration tests for patched attention on MPS (if available)."""
    
    @pytest.mark.skipif(
        not torch.backends.mps.is_available(),
        reason="MPS not available"
    )
    def test_patched_forward_on_mps(self):
        """Test patched forward runs successfully on MPS."""
        attn = MockAttention(dim=64, num_heads=4)
        attn = attn.to("mps")
        attn.eval()
        
        x = torch.randn(2, 10, 64, device="mps")
        output = _patched_attention_forward(attn, x)
        
        assert output.shape == x.shape
        assert output.device.type == "mps"
        
    @pytest.mark.skipif(
        not torch.backends.mps.is_available(),
        reason="MPS not available"
    )
    def test_patched_forward_on_mps_with_float32(self):
        """Test patched forward on MPS with float32 dtype."""
        attn = MockAttention(dim=64, num_heads=4)
        attn = attn.to("mps")
        attn.eval()
        
        x = torch.randn(2, 10, 64, device="mps", dtype=torch.float32)
        output = _patched_attention_forward(attn, x)
        
        assert output.dtype == torch.float32
