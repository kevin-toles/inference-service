"""Model Manager Service for managing model lifecycle.

This module provides centralized model management including:
- Loading/unloading models from configuration presets
- Tracking loaded and available models
- Enforcing memory limits
- Supporting concurrent model access
- Role-based model lookup

Reference: WBS-INF6 AC-6.1 through AC-6.5
Follows: AP-10.1 (asyncio.Lock for concurrent access)
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.providers.llamacpp import LlamaCppProvider


# =============================================================================
# Exceptions (AP-7: All exception class names end in 'Error')
# =============================================================================


class ModelManagerError(Exception):
    """Base exception for ModelManager errors."""

    pass


class ModelNotAvailableError(ModelManagerError):
    """Raised when a model is not available in the configuration."""

    pass


class ModelNotLoadedError(ModelManagerError):
    """Raised when trying to access a model that is not loaded."""

    pass


class MemoryLimitExceededError(ModelManagerError):
    """Raised when loading a model would exceed memory limits."""

    pass


# =============================================================================
# Data Models
# =============================================================================


@dataclass
class ModelInfo:
    """Information about a model."""

    model_id: str
    name: str = ""
    file: str = ""
    size_gb: float = 0.0
    context_length: int = 0
    roles: list[str] = field(default_factory=list)
    status: str = "available"


@dataclass
class PresetLoadResult:
    """Result of loading a configuration preset."""

    preset_id: str
    models_loaded: list[str]
    orchestration_mode: str
    total_memory_gb: float


# =============================================================================
# ModelManager Service
# =============================================================================


class ModelManager:
    """Centralized model lifecycle management service.

    Manages model loading/unloading, memory tracking, concurrent access,
    and role-based model lookup.

    Attributes:
        models_dir: Directory containing model files.
        memory_limit_gb: Maximum memory usage in gigabytes.
        current_memory_gb: Current memory usage in gigabytes.
    """

    def __init__(
        self,
        models_dir: Path,
        model_configs: dict[str, Any],
        config_presets: dict[str, Any] | None = None,
        memory_limit_gb: float = 16.0,
    ) -> None:
        """Initialize the ModelManager.

        Args:
            models_dir: Path to the directory containing model files.
            model_configs: Dictionary mapping model IDs to their configurations.
            config_presets: Dictionary mapping preset categories to preset configs.
            memory_limit_gb: Maximum memory usage allowed in GB.
        """
        self.models_dir = models_dir
        self._model_configs = model_configs
        self._config_presets = config_presets or {}
        self.memory_limit_gb = memory_limit_gb
        self.current_memory_gb = 0.0

        # Track loaded models and their providers
        self._loaded_models: dict[str, LlamaCppProvider] = {}
        self._model_locks: dict[str, asyncio.Lock] = {}

        # Global lock for model loading/unloading operations
        self._global_lock = asyncio.Lock()

    # =========================================================================
    # Model Availability
    # =========================================================================

    def get_available_models(self) -> list[str]:
        """Get list of available model IDs.

        Returns:
            List of model IDs that are configured and have files present.
        """
        available = []
        for model_id in self._model_configs:
            model_path = self._get_model_path(model_id)
            if model_path and model_path.exists():
                available.append(model_id)
        return available

    def get_loaded_models(self) -> list[str]:
        """Get list of currently loaded model IDs.

        Returns:
            List of model IDs that are currently loaded into memory.
        """
        return list(self._loaded_models.keys())

    def get_model_status(self, model_id: str) -> str:
        """Get the status of a model.

        Args:
            model_id: The model identifier.

        Returns:
            Status string: 'loaded', 'available', or 'unknown'.
        """
        if model_id in self._loaded_models:
            return "loaded"
        if model_id in self._model_configs:
            return "available"
        return "unknown"

    def get_model_info(self, model_id: str) -> ModelInfo:
        """Get information about a model.

        Args:
            model_id: The model identifier.

        Returns:
            ModelInfo dataclass with model details.

        Raises:
            ModelNotAvailableError: If model is not in configuration.
        """
        if model_id not in self._model_configs:
            msg = f"Model '{model_id}' is not available in configuration"
            raise ModelNotAvailableError(msg)

        config = self._model_configs[model_id]
        return ModelInfo(
            model_id=model_id,
            name=config.get("name", model_id),
            file=config.get("file", ""),
            size_gb=config.get("size_gb", 0.0),
            context_length=config.get("context_length", 0),
            roles=config.get("roles", []),
            status=self.get_model_status(model_id),
        )

    def list_all_models(self) -> list[ModelInfo]:
        """List all models with their status.

        Returns:
            List of ModelInfo objects for all configured models.
        """
        return [self.get_model_info(model_id) for model_id in self._model_configs]

    # =========================================================================
    # Model Loading/Unloading
    # =========================================================================

    async def load_model(self, model_id: str) -> None:
        """Load a model into memory.

        Args:
            model_id: The model identifier to load.

        Raises:
            ModelNotAvailableError: If model is not available.
            MemoryLimitExceededError: If loading would exceed memory limit.
        """
        async with self._global_lock:
            # Already loaded - idempotent operation
            if model_id in self._loaded_models:
                return

            # Check model exists
            if model_id not in self._model_configs:
                msg = f"Model '{model_id}' is not available in configuration"
                raise ModelNotAvailableError(msg)

            # Check memory limit
            config = self._model_configs[model_id]
            model_size = config.get("size_gb", 0.0)
            if self.current_memory_gb + model_size > self.memory_limit_gb:
                msg = (
                    f"Loading '{model_id}' ({model_size:.1f}GB) would exceed "
                    f"memory limit ({self.current_memory_gb:.1f}GB + "
                    f"{model_size:.1f}GB > {self.memory_limit_gb:.1f}GB)"
                )
                raise MemoryLimitExceededError(msg)

            # Get model path
            model_path = self._get_model_path(model_id)
            if not model_path or not model_path.exists():
                msg = f"Model file not found for '{model_id}'"
                raise ModelNotAvailableError(msg)

            # Create and load provider
            provider = LlamaCppProvider(
                model_path=model_path,
                model_id=model_id,
                context_length=config.get("context_length", 2048),
                n_gpu_layers=-1,  # Metal acceleration on Mac (AP-5.5)
            )
            await provider.load()

            # Track loaded model
            self._loaded_models[model_id] = provider
            self._model_locks[model_id] = asyncio.Lock()
            self.current_memory_gb += model_size

    async def unload_model(self, model_id: str) -> None:
        """Unload a model from memory.

        Args:
            model_id: The model identifier to unload.
        """
        async with self._global_lock:
            if model_id not in self._loaded_models:
                return  # No-op if not loaded

            # Get memory to free
            config = self._model_configs.get(model_id, {})
            model_size = config.get("size_gb", 0.0)

            # Unload provider
            provider = self._loaded_models[model_id]
            await provider.unload()

            # Clean up tracking
            del self._loaded_models[model_id]
            del self._model_locks[model_id]
            self.current_memory_gb -= model_size
            self.current_memory_gb = max(0.0, self.current_memory_gb)

    def get_provider(self, model_id: str) -> LlamaCppProvider:
        """Get the provider for a loaded model.

        Args:
            model_id: The model identifier.

        Returns:
            The LlamaCppProvider instance for the model.

        Raises:
            ModelNotLoadedError: If model is not currently loaded.
        """
        if model_id not in self._loaded_models:
            msg = f"Model '{model_id}' is not loaded"
            raise ModelNotLoadedError(msg)

        return self._loaded_models[model_id]

    # =========================================================================
    # Preset Loading
    # =========================================================================

    async def load_preset(self, preset_id: str) -> PresetLoadResult:
        """Load models from a configuration preset.

        Args:
            preset_id: The preset identifier (e.g., 'D3', 'S1').

        Returns:
            PresetLoadResult with loaded models and orchestration mode.

        Raises:
            ModelManagerError: If preset is not found.
        """
        # Find preset in config categories
        preset_config = None
        for _category, presets in self._config_presets.items():
            if preset_id in presets:
                preset_config = presets[preset_id]
                break

        if preset_config is None:
            msg = f"Configuration preset '{preset_id}' not found"
            raise ModelManagerError(msg)

        # Load all models in preset
        models_to_load = preset_config.get("models", [])
        loaded_models = []
        total_memory = 0.0

        for model_id in models_to_load:
            await self.load_model(model_id)
            loaded_models.append(model_id)
            config = self._model_configs.get(model_id, {})
            total_memory += config.get("size_gb", 0.0)

        return PresetLoadResult(
            preset_id=preset_id,
            models_loaded=loaded_models,
            orchestration_mode=preset_config.get("orchestration_mode", "single"),
            total_memory_gb=total_memory,
        )

    # =========================================================================
    # Role-Based Lookup
    # =========================================================================

    def get_model_by_role(self, role: str) -> str | None:
        """Get a loaded model that supports the specified role.

        Args:
            role: The role to look for (e.g., 'thinker', 'coder', 'fast').

        Returns:
            Model ID if found, None otherwise.
        """
        for model_id in self._loaded_models:
            config = self._model_configs.get(model_id, {})
            roles = config.get("roles", [])
            if role in roles:
                return model_id
        return None

    def get_models_by_role(self, role: str) -> list[str]:
        """Get all loaded models that support the specified role.

        Args:
            role: The role to look for.

        Returns:
            List of model IDs that support the role.
        """
        models = []
        for model_id in self._loaded_models:
            config = self._model_configs.get(model_id, {})
            roles = config.get("roles", [])
            if role in roles:
                models.append(model_id)
        return models

    # =========================================================================
    # Internal Methods
    # =========================================================================

    def _get_model_path(self, model_id: str) -> Path | None:
        """Get the file path for a model.

        Args:
            model_id: The model identifier.

        Returns:
            Path to the model file, or None if not found.
        """
        if model_id not in self._model_configs:
            return None

        config = self._model_configs[model_id]
        filename = config.get("file", "")
        if not filename:
            return None

        # filename can include subdirectory (e.g., "deepseek-r1-7b/model.gguf")
        model_path: Path = self.models_dir / filename
        return model_path


# =============================================================================
# Singleton Instance
# =============================================================================

_model_manager: ModelManager | None = None


def get_model_manager() -> ModelManager:
    """Get singleton ModelManager instance.

    Auto-initializes from config files on first call.

    Returns:
        The shared ModelManager instance.
    """
    global _model_manager
    if _model_manager is None:
        import yaml

        from src.core.config import get_settings

        settings = get_settings()
        models_dir = Path(settings.models_dir)
        config_dir = Path(settings.config_dir)

        # Load model configs
        models_yaml = config_dir / "models.yaml"
        model_configs: dict[str, Any] = {}
        if models_yaml.exists():
            with open(models_yaml) as f:
                data = yaml.safe_load(f) or {}
                model_configs = data.get("models", {})

        # Load presets
        presets_yaml = config_dir / "presets.yaml"
        config_presets: dict[str, Any] = {}
        if presets_yaml.exists():
            with open(presets_yaml) as f:
                config_presets = yaml.safe_load(f) or {}

        _model_manager = ModelManager(
            models_dir=models_dir,
            model_configs=model_configs,
            config_presets=config_presets,
            memory_limit_gb=16.0,  # Default, override via settings if needed
        )
    return _model_manager