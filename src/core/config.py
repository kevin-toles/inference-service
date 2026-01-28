"""Core configuration module for inference-service.

Loads settings from INFERENCE_* prefixed environment variables using Pydantic Settings.

Patterns applied:
- pydantic-settings BaseSettings (Pydantic v2 split) - Issue #4
- env_prefix = "INFERENCE_" for namespace isolation - Issue #18
- @field_validator + @classmethod (Pydantic v2 pattern)
- @lru_cache for singleton pattern
- PEP 604 union syntax (X | None)

Reference: WBS-INF2 AC-2.1, AC-2.4
"""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings

from src.core.constants import (
    CONTAINER_CONFIG_DIR,
    CONTAINER_MODELS_DIR,
    DEFAULT_GPU_LAYERS,
    DEFAULT_HOST,
    DEFAULT_LOG_LEVEL,
    DEFAULT_PORT,
    DEFAULT_SERVICE_NAME,
)


class Settings(BaseSettings):
    """Application settings loaded from INFERENCE_* environment variables.

    All environment variables must be prefixed with INFERENCE_.
    Example: INFERENCE_PORT=8085, INFERENCE_LOG_LEVEL=DEBUG

    Attributes:
        service_name: Service identifier for logging and discovery.
        port: HTTP port (1-65535). Default: 8085.
        host: Bind address. Default: 0.0.0.0.
        environment: Deployment environment. Default: development.
        log_level: Logging verbosity. Default: INFO.
        models_dir: Path to GGUF model files.
        gpu_layers: Layers on GPU (-1 = all). Default: -1.
        backend: Inference backend. Default: llamacpp.
        orchestration_mode: Multi-model orchestration mode. Default: single.
    """

    # =========================================================================
    # Core Settings
    # =========================================================================
    service_name: str = Field(
        default=DEFAULT_SERVICE_NAME,
        description="Service name for identification",
    )
    port: int = Field(
        default=DEFAULT_PORT,
        ge=1,
        le=65535,
        description="HTTP server port",
    )
    host: str = Field(
        default=DEFAULT_HOST,
        description="HTTP server bind address",
    )
    environment: Literal["development", "staging", "production"] = Field(
        default="development",
        description="Deployment environment",
    )
    log_level: str = Field(
        default=DEFAULT_LOG_LEVEL,
        description="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)",
    )

    # =========================================================================
    # Model Configuration
    # =========================================================================
    models_dir: str = Field(
        default=CONTAINER_MODELS_DIR,
        description="Path to directory containing GGUF model files",
    )
    config_dir: str = Field(
        default=CONTAINER_CONFIG_DIR,
        description="Path to directory containing config files (models.yaml, presets.yaml)",
    )
    gpu_layers: int = Field(
        default=DEFAULT_GPU_LAYERS,
        description="Number of layers to offload to GPU (-1 = all)",
    )
    gpu_index: int = Field(
        default=0,
        ge=0,
        description="GPU device index for multi-GPU systems (sets CUDA_VISIBLE_DEVICES)",
    )
    backend: Literal["llamacpp", "vllm"] = Field(
        default="llamacpp",
        description="Inference backend to use",
    )

    # =========================================================================
    # Orchestration Settings
    # =========================================================================
    orchestration_mode: Literal[
        "single", "critique", "debate", "ensemble", "pipeline"
    ] = Field(
        default="single",
        description="Multi-model orchestration mode",
    )

    # =========================================================================
    # Auto-Load Preset on Startup
    # =========================================================================
    default_preset: str | None = Field(
        default=None,
        description="Preset to auto-load on startup (e.g., 'D4', 'S1'). If not set, no models are loaded.",
    )

    # =========================================================================
    # Vision Language Model (VLM) Configuration - Moondream 2
    # =========================================================================
    vision_model_id: str = Field(
        default="vikhyatk/moondream2",
        description="HuggingFace model ID for the VLM. Set to empty string to disable.",
    )
    vision_model_revision: str = Field(
        default="2025-01-09",
        description="Model revision/version from HuggingFace.",
    )
    vision_context_length: int = Field(
        default=2048,
        ge=512,
        le=8192,
        description="Context window size for VLM (Moondream 2 supports 2048).",
    )
    vision_device: str | None = Field(
        default=None,
        description="Device for VLM inference: 'cpu', 'mps', 'cuda', or None (auto-detect). Moondream runs well on MPS with ~5GB memory.",
    )

    # =========================================================================
    # Testing/Development Options
    # =========================================================================
    skip_path_validation: bool = Field(
        default=False,
        description="Skip path existence validation (for testing only)",
    )

    # =========================================================================
    # Pydantic v2 Model Configuration
    # =========================================================================
    model_config = {
        "env_prefix": "INFERENCE_",
        "case_sensitive": False,
        "extra": "ignore",
    }

    # =========================================================================
    # Validators (Pydantic v2 pattern: @field_validator + @classmethod)
    # =========================================================================
    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        """Validate and normalize log level to uppercase.

        Args:
            v: Input log level string.

        Returns:
            Normalized uppercase log level.

        Raises:
            ValueError: If log level is not valid.
        """
        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        normalized = v.upper()
        if normalized not in valid_levels:
            msg = f"log_level must be one of {valid_levels}, got '{v}'"
            raise ValueError(msg)
        return normalized

    @model_validator(mode="after")
    def validate_paths_exist(self) -> "Settings":
        """Validate that configured paths exist.

        This helps catch misconfiguration early during startup rather than
        failing silently later when trying to load models.

        Skipped when skip_path_validation=True (for testing).

        Returns:
            Self if validation passes.

        Raises:
            ValueError: If models_dir or config_dir does not exist.
        """
        # Allow skipping for tests
        if self.skip_path_validation:
            return self

        models_path = Path(self.models_dir)
        config_path = Path(self.config_dir)

        # Collect all path errors for a single, helpful error message
        errors: list[str] = []

        if not models_path.exists():
            errors.append(
                f"models_dir '{self.models_dir}' does not exist. "
                f"Set INFERENCE_MODELS_DIR or mount volume to '{self.models_dir}'"
            )

        if not config_path.exists():
            errors.append(
                f"config_dir '{self.config_dir}' does not exist. "
                f"Set INFERENCE_CONFIG_DIR or mount volume to '{self.config_dir}'"
            )

        if errors:
            msg = "Path configuration error(s):\n  - " + "\n  - ".join(errors)
            raise ValueError(msg)

        return self


@lru_cache
def get_settings() -> Settings:
    """Get singleton Settings instance.

    Uses @lru_cache to ensure only one instance is created.
    This follows the singleton pattern from CODING_PATTERNS_ANALYSIS.md.

    Returns:
        Cached Settings instance.
    """
    return Settings()
