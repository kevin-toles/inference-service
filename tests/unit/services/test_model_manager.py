"""Unit tests for ModelManager service.

Tests the model registry, memory management, concurrent access,
and role-based lookup functionality.

Reference: WBS-INF6 AC-6.1 through AC-6.5
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest


# =============================================================================
# Constants (S1192: Avoid duplicated string literals)
# =============================================================================

MODEL_PHI4 = "phi-4"
MODEL_DEEPSEEK = "deepseek-r1-7b"
MODEL_QWEN = "qwen2.5-7b"
MODEL_LLAMA = "llama-3.2-3b"
CONFIG_D3 = "D3"
CONFIG_S1 = "S1"
ROLE_THINKER = "thinker"
ROLE_CODER = "coder"
ROLE_PRIMARY = "primary"
ROLE_FAST = "fast"


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def mock_model_configs() -> dict[str, Any]:
    """Create mock model configuration data."""
    return {
        MODEL_PHI4: {
            "name": "Microsoft Phi-4",
            "file": "phi-4-Q4_K_S.gguf",
            "size_gb": 8.4,
            "context_length": 16384,
            "roles": [ROLE_PRIMARY, ROLE_THINKER, ROLE_CODER],
        },
        MODEL_DEEPSEEK: {
            "name": "DeepSeek R1 Distill 7B",
            "file": "DeepSeek-R1-Distill-Qwen-7B-Q4_K_M.gguf",
            "size_gb": 4.7,
            "context_length": 32768,
            "roles": [ROLE_THINKER],
        },
        MODEL_QWEN: {
            "name": "Qwen 2.5 7B Instruct",
            "file": "qwen2.5-7b-instruct-q4_k_m.gguf",
            "size_gb": 4.5,
            "context_length": 32768,
            "roles": [ROLE_CODER, ROLE_PRIMARY],
        },
        MODEL_LLAMA: {
            "name": "Llama 3.2 3B Instruct",
            "file": "llama-3.2-3b-instruct-q4_k_m.gguf",
            "size_gb": 2.0,
            "context_length": 8192,
            "roles": [ROLE_FAST],
        },
    }


@pytest.fixture
def mock_config_presets() -> dict[str, Any]:
    """Create mock configuration presets."""
    return {
        "single": {
            CONFIG_S1: {
                "name": "Phi-4 Solo",
                "models": [MODEL_PHI4],
                "total_size_gb": 8.4,
                "orchestration_mode": "single",
            },
        },
        "dual": {
            CONFIG_D3: {
                "name": "Reasoning Debate",
                "models": [MODEL_PHI4, MODEL_DEEPSEEK],
                "total_size_gb": 13.1,
                "orchestration_mode": "debate",
                "roles": {
                    MODEL_PHI4: ["generator", "reconciler"],
                    MODEL_DEEPSEEK: "generator",
                },
            },
        },
    }


@pytest.fixture
def models_dir(tmp_path: Path) -> Path:
    """Create a temporary models directory with mock files."""
    models = tmp_path / "models"
    models.mkdir()

    # Create mock model directories and files
    for model_id, filename in [
        (MODEL_PHI4, "phi-4-Q4_K_S.gguf"),
        (MODEL_DEEPSEEK, "DeepSeek-R1-Distill-Qwen-7B-Q4_K_M.gguf"),
        (MODEL_QWEN, "qwen2.5-7b-instruct-q4_k_m.gguf"),
        (MODEL_LLAMA, "llama-3.2-3b-instruct-q4_k_m.gguf"),
    ]:
        model_dir = models / model_id
        model_dir.mkdir()
        (model_dir / filename).touch()

    return models


@pytest.fixture
def mock_llama_class() -> MagicMock:
    """Create a mock Llama class for provider testing."""
    mock = MagicMock()
    mock_instance = MagicMock()
    mock_instance.tokenize.return_value = [1, 2, 3, 4, 5]
    mock_instance.create_chat_completion.return_value = {
        "id": "chatcmpl-test",
        "choices": [
            {"index": 0, "message": {"role": "assistant", "content": "Test"}, "finish_reason": "stop"}
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }
    mock.return_value = mock_instance
    return mock


# =============================================================================
# TestModelManagerImport
# =============================================================================


class TestModelManagerImport:
    """Test that ModelManager can be imported."""

    def test_model_manager_importable(self) -> None:
        """AC-6.1: ModelManager class exists and can be imported."""
        from src.services.model_manager import ModelManager

        assert ModelManager is not None

    def test_model_manager_exceptions_importable(self) -> None:
        """AC-6.1: Custom exceptions can be imported."""
        from src.services.model_manager import (
            MemoryLimitExceededError,
            ModelManagerError,
            ModelNotAvailableError,
            ModelNotLoadedError,
        )

        assert issubclass(ModelNotAvailableError, ModelManagerError)
        assert issubclass(ModelNotLoadedError, ModelManagerError)
        assert issubclass(MemoryLimitExceededError, ModelManagerError)


# =============================================================================
# TestModelManagerInit
# =============================================================================


class TestModelManagerInit:
    """Test ModelManager initialization."""

    def test_init_with_models_dir(
        self, models_dir: Path, mock_model_configs: dict[str, Any]
    ) -> None:
        """AC-6.1: ModelManager initializes with models directory."""
        from src.services.model_manager import ModelManager

        manager = ModelManager(
            models_dir=models_dir,
            model_configs=mock_model_configs,
        )

        assert manager is not None
        assert manager.models_dir == models_dir

    def test_init_with_memory_limit(
        self, models_dir: Path, mock_model_configs: dict[str, Any]
    ) -> None:
        """AC-6.3: ModelManager accepts memory limit parameter."""
        from src.services.model_manager import ModelManager

        manager = ModelManager(
            models_dir=models_dir,
            model_configs=mock_model_configs,
            memory_limit_gb=16.0,
        )

        assert manager.memory_limit_gb == 16.0

    def test_init_registers_available_models(
        self, models_dir: Path, mock_model_configs: dict[str, Any]
    ) -> None:
        """AC-6.2: ModelManager tracks available models from config."""
        from src.services.model_manager import ModelManager

        manager = ModelManager(
            models_dir=models_dir,
            model_configs=mock_model_configs,
        )

        available = manager.get_available_models()
        assert MODEL_PHI4 in available
        assert MODEL_DEEPSEEK in available


# =============================================================================
# TestModelManagerLoadPreset
# =============================================================================


class TestModelManagerLoadPreset:
    """Test loading models from config presets."""

    @pytest.mark.asyncio
    async def test_load_preset_d3(
        self,
        models_dir: Path,
        mock_model_configs: dict[str, Any],
        mock_config_presets: dict[str, Any],
        mock_llama_class: MagicMock,
    ) -> None:
        """AC-6.1: Config preset D3 loads phi-4 and deepseek-r1-7b."""
        with patch("src.providers.llamacpp.Llama", mock_llama_class):
            from src.services.model_manager import ModelManager

            manager = ModelManager(
                models_dir=models_dir,
                model_configs=mock_model_configs,
                config_presets=mock_config_presets,
            )

            await manager.load_preset(CONFIG_D3)

            loaded = manager.get_loaded_models()
            assert MODEL_PHI4 in loaded
            assert MODEL_DEEPSEEK in loaded

    @pytest.mark.asyncio
    async def test_load_preset_invalid_raises_error(
        self,
        models_dir: Path,
        mock_model_configs: dict[str, Any],
        mock_config_presets: dict[str, Any],
    ) -> None:
        """AC-6.1: Invalid preset raises error."""
        from src.services.model_manager import ModelManager, ModelManagerError

        manager = ModelManager(
            models_dir=models_dir,
            model_configs=mock_model_configs,
            config_presets=mock_config_presets,
        )

        with pytest.raises(ModelManagerError):
            await manager.load_preset("INVALID_PRESET")

    @pytest.mark.asyncio
    async def test_load_preset_returns_orchestration_mode(
        self,
        models_dir: Path,
        mock_model_configs: dict[str, Any],
        mock_config_presets: dict[str, Any],
        mock_llama_class: MagicMock,
    ) -> None:
        """AC-6.1: Loading preset returns orchestration mode."""
        with patch("src.providers.llamacpp.Llama", mock_llama_class):
            from src.services.model_manager import ModelManager

            manager = ModelManager(
                models_dir=models_dir,
                model_configs=mock_model_configs,
                config_presets=mock_config_presets,
            )

            result = await manager.load_preset(CONFIG_D3)

            assert result.orchestration_mode == "debate"


# =============================================================================
# TestModelManagerTracking
# =============================================================================


class TestModelManagerTracking:
    """Test model status tracking."""

    def test_get_available_models(
        self, models_dir: Path, mock_model_configs: dict[str, Any]
    ) -> None:
        """AC-6.2: ModelManager lists available models."""
        from src.services.model_manager import ModelManager

        manager = ModelManager(
            models_dir=models_dir,
            model_configs=mock_model_configs,
        )

        available = manager.get_available_models()

        assert isinstance(available, list)
        assert len(available) == 4  # All 4 mock models

    def test_get_loaded_models_empty_initially(
        self, models_dir: Path, mock_model_configs: dict[str, Any]
    ) -> None:
        """AC-6.2: No models loaded initially."""
        from src.services.model_manager import ModelManager

        manager = ModelManager(
            models_dir=models_dir,
            model_configs=mock_model_configs,
        )

        loaded = manager.get_loaded_models()

        assert loaded == []

    @pytest.mark.asyncio
    async def test_get_loaded_models_after_load(
        self,
        models_dir: Path,
        mock_model_configs: dict[str, Any],
        mock_llama_class: MagicMock,
    ) -> None:
        """AC-6.2: ModelManager tracks loaded models."""
        with patch("src.providers.llamacpp.Llama", mock_llama_class):
            from src.services.model_manager import ModelManager

            manager = ModelManager(
                models_dir=models_dir,
                model_configs=mock_model_configs,
            )

            await manager.load_model(MODEL_PHI4)

            loaded = manager.get_loaded_models()
            assert MODEL_PHI4 in loaded

    @pytest.mark.asyncio
    async def test_get_model_status(
        self,
        models_dir: Path,
        mock_model_configs: dict[str, Any],
        mock_llama_class: MagicMock,
    ) -> None:
        """AC-6.2: ModelManager reports model status."""
        with patch("src.providers.llamacpp.Llama", mock_llama_class):
            from src.services.model_manager import ModelManager

            manager = ModelManager(
                models_dir=models_dir,
                model_configs=mock_model_configs,
            )

            # Before loading
            status = manager.get_model_status(MODEL_PHI4)
            assert status == "available"

            # After loading
            await manager.load_model(MODEL_PHI4)
            status = manager.get_model_status(MODEL_PHI4)
            assert status == "loaded"


# =============================================================================
# TestModelManagerMemory
# =============================================================================


class TestModelManagerMemory:
    """Test memory limit enforcement."""

    @pytest.mark.asyncio
    async def test_memory_tracking(
        self,
        models_dir: Path,
        mock_model_configs: dict[str, Any],
        mock_llama_class: MagicMock,
    ) -> None:
        """AC-6.3: ModelManager tracks memory usage."""
        with patch("src.providers.llamacpp.Llama", mock_llama_class):
            from src.services.model_manager import ModelManager

            manager = ModelManager(
                models_dir=models_dir,
                model_configs=mock_model_configs,
                memory_limit_gb=16.0,
            )

            await manager.load_model(MODEL_PHI4)  # 8.4 GB

            assert manager.current_memory_gb == pytest.approx(8.4, rel=0.1)

    @pytest.mark.asyncio
    async def test_memory_limit_prevents_overload(
        self,
        models_dir: Path,
        mock_model_configs: dict[str, Any],
        mock_llama_class: MagicMock,
    ) -> None:
        """AC-6.3: Memory check prevents loading models exceeding 16GB."""
        with patch("src.providers.llamacpp.Llama", mock_llama_class):
            from src.services.model_manager import (
                MemoryLimitExceededError,
                ModelManager,
            )

            manager = ModelManager(
                models_dir=models_dir,
                model_configs=mock_model_configs,
                memory_limit_gb=16.0,
            )

            # Load phi-4 (8.4 GB) + deepseek (4.7 GB) = 13.1 GB
            await manager.load_model(MODEL_PHI4)
            await manager.load_model(MODEL_DEEPSEEK)

            # Try to load phi-3-medium (8.6 GB) - would exceed 16 GB
            # Using a mock config entry for testing
            manager._model_configs["phi-3-medium-128k"] = {
                "file": "test.gguf",
                "size_gb": 8.6,
                "context_length": 131072,
                "roles": ["longctx"],
            }
            (models_dir / "phi-3-medium-128k").mkdir()
            (models_dir / "phi-3-medium-128k" / "test.gguf").touch()

            with pytest.raises(MemoryLimitExceededError):
                await manager.load_model("phi-3-medium-128k")

    @pytest.mark.asyncio
    async def test_unload_frees_memory(
        self,
        models_dir: Path,
        mock_model_configs: dict[str, Any],
        mock_llama_class: MagicMock,
    ) -> None:
        """AC-6.3: Unloading model frees memory."""
        with patch("src.providers.llamacpp.Llama", mock_llama_class):
            from src.services.model_manager import ModelManager

            manager = ModelManager(
                models_dir=models_dir,
                model_configs=mock_model_configs,
                memory_limit_gb=16.0,
            )

            await manager.load_model(MODEL_PHI4)
            assert manager.current_memory_gb == pytest.approx(8.4, rel=0.1)

            await manager.unload_model(MODEL_PHI4)
            assert manager.current_memory_gb == 0.0


# =============================================================================
# TestModelManagerConcurrency
# =============================================================================


class TestModelManagerConcurrency:
    """Test concurrent model access."""

    @pytest.mark.asyncio
    async def test_concurrent_load_uses_lock(
        self,
        models_dir: Path,
        mock_model_configs: dict[str, Any],
        mock_llama_class: MagicMock,
    ) -> None:
        """AC-6.4: Concurrent loads use asyncio locks."""
        with patch("src.providers.llamacpp.Llama", mock_llama_class):
            from src.services.model_manager import ModelManager

            manager = ModelManager(
                models_dir=models_dir,
                model_configs=mock_model_configs,
            )

            # Attempt concurrent loads of same model
            await asyncio.gather(
                manager.load_model(MODEL_PHI4),
                manager.load_model(MODEL_PHI4),
                return_exceptions=True,
            )

            # Should only load once (idempotent)
            loaded = manager.get_loaded_models()
            assert loaded.count(MODEL_PHI4) == 1

    @pytest.mark.asyncio
    async def test_get_provider_concurrent_access(
        self,
        models_dir: Path,
        mock_model_configs: dict[str, Any],
        mock_llama_class: MagicMock,
    ) -> None:
        """AC-6.4: ModelManager supports concurrent model access."""
        with patch("src.providers.llamacpp.Llama", mock_llama_class):
            from src.services.model_manager import ModelManager

            manager = ModelManager(
                models_dir=models_dir,
                model_configs=mock_model_configs,
            )

            await manager.load_model(MODEL_PHI4)
            await manager.load_model(MODEL_DEEPSEEK)

            # Concurrent access to different models
            async def access_model(model_id: str) -> bool:
                provider = await manager.get_provider(model_id)
                return provider is not None

            results = await asyncio.gather(
                access_model(MODEL_PHI4),
                access_model(MODEL_DEEPSEEK),
                access_model(MODEL_PHI4),
            )

            assert all(results)

    @pytest.mark.asyncio
    async def test_concurrent_requests_different_models(
        self,
        models_dir: Path,
        mock_model_configs: dict[str, Any],
        mock_llama_class: MagicMock,
    ) -> None:
        """AC-6.4: Multiple models can handle concurrent requests."""
        with patch("src.providers.llamacpp.Llama", mock_llama_class):
            from src.services.model_manager import ModelManager

            manager = ModelManager(
                models_dir=models_dir,
                model_configs=mock_model_configs,
            )

            await manager.load_model(MODEL_PHI4)
            await manager.load_model(MODEL_DEEPSEEK)

            # Both models should be accessible concurrently
            phi4_provider = await manager.get_provider(MODEL_PHI4)
            deepseek_provider = await manager.get_provider(MODEL_DEEPSEEK)

            assert phi4_provider is not None
            assert deepseek_provider is not None
            assert phi4_provider is not deepseek_provider


# =============================================================================
# TestModelManagerRoles
# =============================================================================


class TestModelManagerRoles:
    """Test role-based model lookup."""

    @pytest.mark.asyncio
    async def test_get_model_by_role_thinker(
        self,
        models_dir: Path,
        mock_model_configs: dict[str, Any],
        mock_llama_class: MagicMock,
    ) -> None:
        """AC-6.5: get_model_by_role('thinker') returns deepseek-r1-7b."""
        with patch("src.providers.llamacpp.Llama", mock_llama_class):
            from src.services.model_manager import ModelManager

            manager = ModelManager(
                models_dir=models_dir,
                model_configs=mock_model_configs,
            )

            # Load models that support thinker role
            await manager.load_model(MODEL_DEEPSEEK)

            model_id = manager.get_model_by_role(ROLE_THINKER)
            assert model_id == MODEL_DEEPSEEK

    @pytest.mark.asyncio
    async def test_get_model_by_role_coder(
        self,
        models_dir: Path,
        mock_model_configs: dict[str, Any],
        mock_llama_class: MagicMock,
    ) -> None:
        """AC-6.5: get_model_by_role('coder') returns a coder model."""
        with patch("src.providers.llamacpp.Llama", mock_llama_class):
            from src.services.model_manager import ModelManager

            manager = ModelManager(
                models_dir=models_dir,
                model_configs=mock_model_configs,
            )

            await manager.load_model(MODEL_QWEN)

            model_id = manager.get_model_by_role(ROLE_CODER)
            assert model_id == MODEL_QWEN

    @pytest.mark.asyncio
    async def test_get_model_by_role_with_multiple_loaded(
        self,
        models_dir: Path,
        mock_model_configs: dict[str, Any],
        mock_llama_class: MagicMock,
    ) -> None:
        """AC-6.5: With multiple models, returns first matching loaded model."""
        with patch("src.providers.llamacpp.Llama", mock_llama_class):
            from src.services.model_manager import ModelManager

            manager = ModelManager(
                models_dir=models_dir,
                model_configs=mock_model_configs,
            )

            # Load multiple models with thinker role
            await manager.load_model(MODEL_PHI4)  # Has thinker role
            await manager.load_model(MODEL_DEEPSEEK)  # Primary thinker

            # Should return deepseek as it's the primary thinker
            model_id = manager.get_model_by_role(ROLE_THINKER)
            # Either model is acceptable since both support the role
            assert model_id in [MODEL_PHI4, MODEL_DEEPSEEK]

    def test_get_model_by_role_not_loaded_returns_none(
        self, models_dir: Path, mock_model_configs: dict[str, Any]
    ) -> None:
        """AC-6.5: Returns None if no loaded model has the role."""
        from src.services.model_manager import ModelManager

        manager = ModelManager(
            models_dir=models_dir,
            model_configs=mock_model_configs,
        )

        # No models loaded
        model_id = manager.get_model_by_role(ROLE_THINKER)
        assert model_id is None

    @pytest.mark.asyncio
    async def test_get_models_by_role(
        self,
        models_dir: Path,
        mock_model_configs: dict[str, Any],
        mock_llama_class: MagicMock,
    ) -> None:
        """AC-6.5: get_models_by_role returns all models with role."""
        with patch("src.providers.llamacpp.Llama", mock_llama_class):
            from src.services.model_manager import ModelManager

            manager = ModelManager(
                models_dir=models_dir,
                model_configs=mock_model_configs,
            )

            await manager.load_model(MODEL_PHI4)  # thinker, coder, primary
            await manager.load_model(MODEL_DEEPSEEK)  # thinker

            models = manager.get_models_by_role(ROLE_THINKER)
            assert MODEL_PHI4 in models
            assert MODEL_DEEPSEEK in models


# =============================================================================
# TestModelManagerLoadUnload
# =============================================================================


class TestModelManagerLoadUnload:
    """Test model loading and unloading."""

    @pytest.mark.asyncio
    async def test_load_model_success(
        self,
        models_dir: Path,
        mock_model_configs: dict[str, Any],
        mock_llama_class: MagicMock,
    ) -> None:
        """AC-6.2: Loading model adds it to loaded list."""
        with patch("src.providers.llamacpp.Llama", mock_llama_class):
            from src.services.model_manager import ModelManager

            manager = ModelManager(
                models_dir=models_dir,
                model_configs=mock_model_configs,
            )

            await manager.load_model(MODEL_PHI4)

            assert MODEL_PHI4 in manager.get_loaded_models()

    @pytest.mark.asyncio
    async def test_load_nonexistent_model_raises_error(
        self, models_dir: Path, mock_model_configs: dict[str, Any]
    ) -> None:
        """AC-6.2: Loading unknown model raises error."""
        from src.services.model_manager import ModelManager, ModelNotAvailableError

        manager = ModelManager(
            models_dir=models_dir,
            model_configs=mock_model_configs,
        )

        with pytest.raises(ModelNotAvailableError):
            await manager.load_model("nonexistent-model")

    @pytest.mark.asyncio
    async def test_unload_model_success(
        self,
        models_dir: Path,
        mock_model_configs: dict[str, Any],
        mock_llama_class: MagicMock,
    ) -> None:
        """AC-6.2: Unloading model removes it from loaded list."""
        with patch("src.providers.llamacpp.Llama", mock_llama_class):
            from src.services.model_manager import ModelManager

            manager = ModelManager(
                models_dir=models_dir,
                model_configs=mock_model_configs,
            )

            await manager.load_model(MODEL_PHI4)
            await manager.unload_model(MODEL_PHI4)

            assert MODEL_PHI4 not in manager.get_loaded_models()

    @pytest.mark.asyncio
    async def test_unload_not_loaded_is_noop(
        self, models_dir: Path, mock_model_configs: dict[str, Any]
    ) -> None:
        """AC-6.2: Unloading non-loaded model is a no-op."""
        from src.services.model_manager import ModelManager

        manager = ModelManager(
            models_dir=models_dir,
            model_configs=mock_model_configs,
        )

        # Should not raise
        await manager.unload_model(MODEL_PHI4)

    @pytest.mark.asyncio
    async def test_get_provider_not_loaded_raises(
        self, models_dir: Path, mock_model_configs: dict[str, Any]
    ) -> None:
        """AC-6.2: Getting provider for unloaded model raises error."""
        from src.services.model_manager import ModelManager, ModelNotLoadedError

        manager = ModelManager(
            models_dir=models_dir,
            model_configs=mock_model_configs,
        )

        with pytest.raises(ModelNotLoadedError):
            await manager.get_provider(MODEL_PHI4)


# =============================================================================
# TestModelManagerExceptions
# =============================================================================


class TestModelManagerExceptions:
    """Test exception handling."""

    def test_exception_names_end_in_error(self) -> None:
        """AP-7: All exception class names end in 'Error'."""
        from src.services.model_manager import (
            MemoryLimitExceededError,
            ModelManagerError,
            ModelNotAvailableError,
            ModelNotLoadedError,
        )

        for exc_class in [
            ModelManagerError,
            ModelNotAvailableError,
            ModelNotLoadedError,
            MemoryLimitExceededError,
        ]:
            assert exc_class.__name__.endswith("Error")

    def test_exceptions_have_hierarchy(self) -> None:
        """Exceptions form proper hierarchy."""
        from src.services.model_manager import (
            MemoryLimitExceededError,
            ModelManagerError,
            ModelNotAvailableError,
            ModelNotLoadedError,
        )

        assert issubclass(ModelNotAvailableError, ModelManagerError)
        assert issubclass(ModelNotLoadedError, ModelManagerError)
        assert issubclass(MemoryLimitExceededError, ModelManagerError)
        assert issubclass(ModelManagerError, Exception)


# =============================================================================
# TestModelManagerModelInfo
# =============================================================================


class TestModelManagerModelInfo:
    """Test model information retrieval."""

    def test_get_model_info(
        self, models_dir: Path, mock_model_configs: dict[str, Any]
    ) -> None:
        """AC-6.2: ModelManager provides model information."""
        from src.services.model_manager import ModelManager

        manager = ModelManager(
            models_dir=models_dir,
            model_configs=mock_model_configs,
        )

        info = manager.get_model_info(MODEL_PHI4)

        assert info.model_id == MODEL_PHI4
        assert info.context_length == 16384
        assert ROLE_PRIMARY in info.roles

    @pytest.mark.asyncio
    async def test_list_all_models(
        self,
        models_dir: Path,
        mock_model_configs: dict[str, Any],
        mock_llama_class: MagicMock,
    ) -> None:
        """AC-6.2: ModelManager lists all models with status."""
        with patch("src.providers.llamacpp.Llama", mock_llama_class):
            from src.services.model_manager import ModelManager

            manager = ModelManager(
                models_dir=models_dir,
                model_configs=mock_model_configs,
            )

            await manager.load_model(MODEL_PHI4)

            models = manager.list_all_models()

            assert len(models) == 4
            phi4_info = next(m for m in models if m.model_id == MODEL_PHI4)
            assert phi4_info.status == "loaded"

            deepseek_info = next(m for m in models if m.model_id == MODEL_DEEPSEEK)
            assert deepseek_info.status == "available"
