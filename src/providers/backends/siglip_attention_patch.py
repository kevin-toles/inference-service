"""Patch for SigLIP ViT attention to work without xformers on MPS.

The DeepSeek-VL2 SigLIP vision encoder has a hardcoded import of xformers
in its Attention.forward() method. Since xformers requires CUDA and cannot
be installed on macOS/MPS, we need to monkey-patch the attention to use
standard PyTorch scaled_dot_product_attention instead.

This patch should be applied BEFORE loading the DeepSeek-VL2 model.

References:
- DeepSeek-VL2 siglip_vit.py: Attention class
- PyTorch SDPA: torch.nn.functional.scaled_dot_product_attention
"""

import logging
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

if TYPE_CHECKING:
    from torch import Tensor

logger = logging.getLogger(__name__)

# Flag to track if patch has been applied
_patch_applied = False


def _patched_attention_forward(self, x: "Tensor") -> "Tensor":
    """Patched forward method that uses PyTorch SDPA instead of xformers.
    
    This is a drop-in replacement for the original Attention.forward() method
    in deepseek_vl2/models/siglip_vit.py. It removes the xformers dependency
    and uses torch.nn.functional.scaled_dot_product_attention instead.
    
    Args:
        self: The Attention module instance
        x: Input tensor of shape (B, N, C)
        
    Returns:
        Output tensor of shape (B, N, C)
    """
    B, N, C = x.shape
    qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
    
    if not self.qk_norm:
        # Without qk_norm: use standard SDPA
        # qkv shape: (B, N, 3, num_heads, head_dim)
        # Need to permute to (3, B, num_heads, N, head_dim) for unbind
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, num_heads, N, head_dim)
        q, k, v = qkv.unbind(0)  # Each is (B, num_heads, N, head_dim)
        
        # Use PyTorch's scaled_dot_product_attention
        x = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.attn_drop.p if self.training else 0.,
        )
        
        # Reshape back: (B, num_heads, N, head_dim) -> (B, N, C)
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
    
    # With qk_norm: apply normalization then SDPA
    qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, num_heads, N, head_dim)
    q, k, v = qkv.unbind(0)
    q, k = self.q_norm(q), self.k_norm(k)
    
    if self.fused_attn:
        # Use SDPA (works on MPS)
        x = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.attn_drop.p if self.training else 0.,
        )
    else:
        # Manual attention computation
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = attn @ v
    
    x = x.transpose(1, 2).reshape(B, N, C)
    x = self.proj(x)
    x = self.proj_drop(x)
    return x


def apply_siglip_attention_patch() -> bool:
    """Apply the attention patch to the SigLIP ViT module.
    
    This function monkey-patches the Attention class in deepseek_vl2.models.siglip_vit
    to use our patched forward method that doesn't require xformers.
    
    Returns:
        True if patch was applied successfully, False otherwise.
        
    Note:
        This function is idempotent - calling it multiple times has no effect
        after the first successful application.
    """
    global _patch_applied
    
    if _patch_applied:
        logger.debug("SigLIP attention patch already applied, skipping")
        return True
    
    try:
        # Import the siglip_vit module
        from deepseek_vl2.models import siglip_vit
        
        # Verify the Attention class exists
        if not hasattr(siglip_vit, 'Attention'):
            logger.error("SigLIP Attention class not found in deepseek_vl2.models.siglip_vit")
            return False
        
        # Store original forward for potential restoration
        original_forward = siglip_vit.Attention.forward
        
        # Apply the patch
        siglip_vit.Attention.forward = _patched_attention_forward
        
        _patch_applied = True
        logger.info(
            "Applied SigLIP attention patch",
            extra={
                "original_method": f"{original_forward.__module__}.{original_forward.__qualname__}",
                "patched_method": f"{_patched_attention_forward.__module__}.{_patched_attention_forward.__qualname__}",
            }
        )
        return True
        
    except ImportError as e:
        logger.warning(
            "Could not import deepseek_vl2.models.siglip_vit for patching",
            extra={"error": str(e)}
        )
        return False
    except Exception as e:
        logger.error(
            "Failed to apply SigLIP attention patch",
            extra={"error": str(e), "error_type": type(e).__name__}
        )
        return False


def is_patch_applied() -> bool:
    """Check if the SigLIP attention patch has been applied.
    
    Returns:
        True if the patch has been applied, False otherwise.
    """
    return _patch_applied
