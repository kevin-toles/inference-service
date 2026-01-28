"""Integration tests for DeepSeek VL2 on MPS with dtype compatibility.

WBS-VLM6: Integration Testing

These tests validate the full integration of the DeviceBackend pattern
with DeepSeek-VL2-Tiny on Apple Silicon MPS.

Acceptance Criteria:
- AC-VLM6.1: DeepSeek-VL2-Tiny loads on MPS without RuntimeError
- AC-VLM6.3: Image classification returns valid result

Note: These tests require the actual model to be present and will be
skipped if the model is not available or MPS is not supported.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

# Skip all tests in this module if MPS is not available
pytestmark = pytest.mark.skipif(
    not torch.backends.mps.is_available(),
    reason="MPS not available on this system",
)


# Model path - adjust based on your setup
MODEL_PATH = Path(
    os.environ.get(
        "DEEPSEEK_VL2_MODEL_PATH",
        "/Users/kevintoles/POC/ai-models/models/deepseek-vl2-tiny",
    )
)


class TestDeepSeekVLMPSIntegration:
    """Integration tests for DeepSeek VL on MPS."""

    @pytest.fixture
    def model_available(self) -> bool:
        """Check if model is available."""
        return MODEL_PATH.exists()

    def test_model_loads_on_mps_without_error(
        self,
        model_available: bool,
    ) -> None:
        """AC-VLM6.1: DeepSeek-VL2-Tiny loads on MPS without RuntimeError.
        
        This is the critical test - if the dtype conversion works correctly,
        the model should load without the bfloat16 error.
        """
        if not model_available:
            pytest.skip(f"Model not found at {MODEL_PATH}")

        from src.providers.deepseek_vl import DeepSeekVLProvider

        # This should NOT raise:
        # RuntimeError: Input tensor is bfloat16 dtype but MPS input 
        # only supports float or float16 dtypes.
        provider = DeepSeekVLProvider(
            model_path=MODEL_PATH,
            model_id="deepseek-vl2-tiny",
            device="mps",
        )

        # Verify backend was created
        assert provider._backend is not None
        
        # Load the model (this is where the error would occur)
        import asyncio
        asyncio.run(provider.load())

        # Verify model loaded successfully
        assert provider.is_loaded
        assert provider._model is not None

        # Verify all parameters are float32 (not bfloat16)
        for name, param in provider._model.named_parameters():
            assert param.dtype != torch.bfloat16, (
                f"Parameter {name} is still bfloat16 after conversion"
            )
            # MPS supports float32 and float16
            assert param.dtype in (torch.float32, torch.float16), (
                f"Parameter {name} has unsupported dtype {param.dtype}"
            )

        # Clean up
        asyncio.run(provider.unload())

    def test_model_on_correct_device(
        self,
        model_available: bool,
    ) -> None:
        """Verify model is moved to MPS device after loading."""
        if not model_available:
            pytest.skip(f"Model not found at {MODEL_PATH}")

        from src.providers.deepseek_vl import DeepSeekVLProvider

        provider = DeepSeekVLProvider(
            model_path=MODEL_PATH,
            model_id="deepseek-vl2-tiny",
            device="mps",
        )

        import asyncio
        asyncio.run(provider.load())

        # Check that parameters are on MPS
        for name, param in provider._model.named_parameters():
            assert param.device.type == "mps", (
                f"Parameter {name} is on {param.device}, expected mps"
            )
            break  # Just check first parameter

        asyncio.run(provider.unload())


class TestDeepSeekVLImageClassification:
    """Integration tests for image classification."""

    @pytest.fixture
    def model_available(self) -> bool:
        """Check if model is available."""
        return MODEL_PATH.exists()

    @pytest.fixture
    def test_image_path(self) -> Path:
        """Get a test image path."""
        # Use an image from the books collection
        test_img = Path(
            "/Users/kevintoles/POC/ai-platform-data/books/images/"
            "20-python-libraries-you-arent-using-but-should/page0002_img00.png"
        )
        return test_img

    def test_image_classification_returns_result(
        self,
        model_available: bool,
        test_image_path: Path,
    ) -> None:
        """AC-VLM6.3: Image classification returns valid result."""
        if not model_available:
            pytest.skip(f"Model not found at {MODEL_PATH}")
        if not test_image_path.exists():
            pytest.skip(f"Test image not found at {test_image_path}")

        from src.providers.deepseek_vl import (
            DeepSeekVLProvider,
            VisionClassifyRequest,
        )

        provider = DeepSeekVLProvider(
            model_path=MODEL_PATH,
            model_id="deepseek-vl2-tiny",
            device="mps",
        )

        import asyncio

        async def run_classification():
            async with provider:
                request = VisionClassifyRequest(
                    image_path=str(test_image_path),
                    max_tokens=50,
                    temperature=0.1,
                )
                result = await provider.classify_image(request)
                return result

        result = asyncio.run(run_classification())

        # Verify result structure
        assert result is not None
        assert result.classification in ("technical", "discard")
        assert result.raw_response is not None
        assert len(result.raw_response) > 0
        assert result.model_id == "deepseek-vl2-tiny"


class TestMPSBackendDirectly:
    """Direct tests of MPSBackend dtype conversion."""

    def test_mps_backend_converts_bfloat16_model(self) -> None:
        """Verify MPSBackend converts a bfloat16 model correctly."""
        from src.providers.backends import MPSBackend
        import torch.nn as nn

        # Create a simple model with bfloat16 weights
        model = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 5),
        )

        # Convert to bfloat16
        model = model.to(torch.bfloat16)

        # Verify it's bfloat16
        for param in model.parameters():
            assert param.dtype == torch.bfloat16

        # Apply backend conversion
        backend = MPSBackend()
        model = backend.ensure_dtype_compatibility(model)

        # Verify all parameters are now float32
        for name, param in model.named_parameters():
            assert param.dtype == torch.float32, (
                f"Parameter {name} is {param.dtype}, expected float32"
            )

        # Verify we can move to MPS without error
        if torch.backends.mps.is_available():
            model = model.to("mps")
            for param in model.parameters():
                assert param.device.type == "mps"
