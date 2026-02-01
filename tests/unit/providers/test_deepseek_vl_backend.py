"""Tests for DeepSeekVLProvider DeviceBackend integration.

TDD RED Phase - WBS-VLM5: Integrate with DeepSeekVLProvider

Acceptance Criteria:
- AC-VLM5.1: DeepSeekVLProvider.__init__ accepts optional backend parameter
- AC-VLM5.2: Default backend created via DeviceBackendFactory.create_backend(device)
- AC-VLM5.3: ensure_dtype_compatibility() called after from_pretrained()
- AC-VLM5.4: ensure_dtype_compatibility() called before .to(device)
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, Mock, patch, call

import pytest
import torch
import torch.nn as nn

from src.providers.backends import DeviceBackend, DeviceBackendFactory, MPSBackend


# We need to test the provider's integration with backends
# without loading the actual heavy model
class TestDeepSeekVLProviderBackendParameter:
    """Tests for AC-VLM5.1: Provider accepts backend parameter."""

    @pytest.fixture
    def mock_model_path(self, tmp_path: Path) -> Path:
        """Create a mock model directory."""
        model_dir = tmp_path / "deepseek-vl2-tiny"
        model_dir.mkdir()
        return model_dir

    def test_provider_accepts_backend_parameter(
        self,
        mock_model_path: Path,
    ) -> None:
        """DeepSeekVLProvider accepts optional backend parameter."""
        from src.providers.deepseek_vl import DeepSeekVLProvider

        # Create a mock backend
        mock_backend = Mock(spec=DeviceBackend)

        # Should accept backend parameter without error
        provider = DeepSeekVLProvider(
            model_path=mock_model_path,
            model_id="test-model",
            backend=mock_backend,
        )

        # Verify backend was stored
        assert provider._backend is mock_backend

    def test_provider_accepts_none_backend(
        self,
        mock_model_path: Path,
    ) -> None:
        """DeepSeekVLProvider accepts None for backend (will use default)."""
        from src.providers.deepseek_vl import DeepSeekVLProvider

        provider = DeepSeekVLProvider(
            model_path=mock_model_path,
            model_id="test-model",
            backend=None,
        )

        # Backend should be set (created from factory)
        assert provider._backend is not None

    def test_provider_backend_has_correct_type(
        self,
        mock_model_path: Path,
    ) -> None:
        """Provider's backend is a DeviceBackend instance."""
        from src.providers.deepseek_vl import DeepSeekVLProvider

        provider = DeepSeekVLProvider(
            model_path=mock_model_path,
            model_id="test-model",
            device="mps",
        )

        assert isinstance(provider._backend, DeviceBackend)


class TestDeepSeekVLProviderDefaultBackend:
    """Tests for AC-VLM5.2: Default backend from factory."""

    @pytest.fixture
    def mock_model_path(self, tmp_path: Path) -> Path:
        """Create a mock model directory."""
        model_dir = tmp_path / "deepseek-vl2-tiny"
        model_dir.mkdir()
        return model_dir

    def test_default_backend_created_for_mps(
        self,
        mock_model_path: Path,
    ) -> None:
        """Default backend is MPSBackend when device='mps'."""
        from src.providers.deepseek_vl import DeepSeekVLProvider

        provider = DeepSeekVLProvider(
            model_path=mock_model_path,
            model_id="test-model",
            device="mps",
        )

        assert isinstance(provider._backend, MPSBackend)

    def test_factory_called_with_device(
        self,
        mock_model_path: Path,
    ) -> None:
        """DeviceBackendFactory.create_backend() called with device string."""
        from src.providers.deepseek_vl import DeepSeekVLProvider

        with patch.object(
            DeviceBackendFactory,
            "create_backend",
            wraps=DeviceBackendFactory.create_backend,
        ) as mock_create:
            _provider = DeepSeekVLProvider(
                model_path=mock_model_path,
                model_id="test-model",
                device="mps",
            )

            mock_create.assert_called_once_with("mps")

    def test_custom_backend_overrides_factory(
        self,
        mock_model_path: Path,
    ) -> None:
        """Custom backend is used instead of factory-created one."""
        from src.providers.deepseek_vl import DeepSeekVLProvider

        custom_backend = Mock(spec=DeviceBackend)

        with patch.object(DeviceBackendFactory, "create_backend") as mock_create:
            provider = DeepSeekVLProvider(
                model_path=mock_model_path,
                model_id="test-model",
                device="mps",
                backend=custom_backend,
            )

            # Factory should NOT be called when custom backend provided
            mock_create.assert_not_called()
            assert provider._backend is custom_backend


class TestDeepSeekVLProviderDtypeConversion:
    """Tests for AC-VLM5.3 and AC-VLM5.4: Conversion timing."""

    @pytest.fixture
    def mock_model_path(self, tmp_path: Path) -> Path:
        """Create a mock model directory."""
        model_dir = tmp_path / "deepseek-vl2-tiny"
        model_dir.mkdir()
        return model_dir

    def test_ensure_called_after_from_pretrained(
        self,
        mock_model_path: Path,
    ) -> None:
        """ensure_dtype_compatibility() called after from_pretrained()."""
        from src.providers.deepseek_vl import DeepSeekVLProvider

        # Create mock backend that tracks calls
        mock_backend = Mock(spec=DeviceBackend)
        mock_backend.ensure_dtype_compatibility = Mock(side_effect=lambda m: m)

        provider = DeepSeekVLProvider(
            model_path=mock_model_path,
            model_id="test-model",
            device="mps",
            backend=mock_backend,
        )

        # Create mock model and patch the imports
        mock_model = MagicMock(spec=nn.Module)
        mock_model.to = Mock(return_value=mock_model)
        mock_model.eval = Mock(return_value=mock_model)

        call_order = []

        def track_from_pretrained(*args, **kwargs):
            call_order.append("from_pretrained")
            return mock_model

        def track_ensure_dtype(*args, **kwargs):
            call_order.append("ensure_dtype_compatibility")
            return args[0]

        mock_backend.ensure_dtype_compatibility.side_effect = track_ensure_dtype

        # Mock the entire transformers module import inside _load_model_sync
        mock_transformers = MagicMock()
        mock_transformers.AutoModelForCausalLM.from_pretrained = track_from_pretrained
        
        mock_deepseek = MagicMock()
        
        with patch.dict("sys.modules", {
            "transformers": mock_transformers,
            "deepseek_vl2": MagicMock(),
            "deepseek_vl2.models": mock_deepseek,
        }):
            provider._load_model_sync()

        # Verify from_pretrained called before ensure_dtype_compatibility
        assert "from_pretrained" in call_order
        assert "ensure_dtype_compatibility" in call_order
        assert call_order.index("from_pretrained") < call_order.index(
            "ensure_dtype_compatibility"
        )

    def test_ensure_called_before_to_device(
        self,
        mock_model_path: Path,
    ) -> None:
        """ensure_dtype_compatibility() called before .to(device)."""
        from src.providers.deepseek_vl import DeepSeekVLProvider

        mock_backend = Mock(spec=DeviceBackend)

        provider = DeepSeekVLProvider(
            model_path=mock_model_path,
            model_id="test-model",
            device="mps",
            backend=mock_backend,
        )

        # Track call order
        call_order = []
        mock_model = MagicMock(spec=nn.Module)

        def track_to(*args, **kwargs):
            call_order.append(("to", args, kwargs))
            return mock_model

        def track_ensure(model):
            call_order.append(("ensure_dtype_compatibility", model))
            return model

        mock_model.to = track_to
        mock_model.eval = Mock(return_value=mock_model)
        mock_backend.ensure_dtype_compatibility.side_effect = track_ensure

        mock_transformers = MagicMock()
        mock_transformers.AutoModelForCausalLM.from_pretrained = Mock(return_value=mock_model)
        
        with patch.dict("sys.modules", {
            "transformers": mock_transformers,
            "deepseek_vl2": MagicMock(),
            "deepseek_vl2.models": MagicMock(),
        }):
            provider._load_model_sync()

        # Find indices
        ensure_idx = next(
            i for i, c in enumerate(call_order) 
            if c[0] == "ensure_dtype_compatibility"
        )
        to_device_idx = next(
            i for i, c in enumerate(call_order)
            if c[0] == "to" and "mps" in str(c)
        )

        # ensure_dtype_compatibility must be called before .to(device)
        assert ensure_idx < to_device_idx

    def test_backend_receives_model_from_pretrained(
        self,
        mock_model_path: Path,
    ) -> None:
        """Backend receives the model returned by from_pretrained()."""
        from src.providers.deepseek_vl import DeepSeekVLProvider

        mock_backend = Mock(spec=DeviceBackend)
        mock_backend.ensure_dtype_compatibility = Mock(side_effect=lambda m: m)

        provider = DeepSeekVLProvider(
            model_path=mock_model_path,
            model_id="test-model",
            device="mps",
            backend=mock_backend,
        )

        mock_model = MagicMock(spec=nn.Module)
        mock_model.to = Mock(return_value=mock_model)
        mock_model.eval = Mock(return_value=mock_model)

        mock_transformers = MagicMock()
        mock_transformers.AutoModelForCausalLM.from_pretrained = Mock(return_value=mock_model)
        
        with patch.dict("sys.modules", {
            "transformers": mock_transformers,
            "deepseek_vl2": MagicMock(),
            "deepseek_vl2.models": MagicMock(),
        }):
            provider._load_model_sync()

        # Backend should have received the mock model
        mock_backend.ensure_dtype_compatibility.assert_called_once_with(mock_model)
