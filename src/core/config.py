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
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings


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
        default="inference-service",
        description="Service name for identification",
    )
    port: int = Field(
        default=8085,
        ge=1,
        le=65535,
        description="HTTP server port",
    )
    host: str = Field(
        default="0.0.0.0",
        description="HTTP server bind address",
    )
    environment: Literal["development", "staging", "production"] = Field(
        default="development",
        description="Deployment environment",
    )
    log_level: str = Field(
        default="INFO",
        description="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)",
    )

    # =========================================================================
    # Model Configuration
    # =========================================================================
    models_dir: str = Field(
        default="/models",
        description="Path to directory containing GGUF model files",
    )
    gpu_layers: int = Field(
        default=-1,
        description="Number of layers to offload to GPU (-1 = all)",
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


@lru_cache
def get_settings() -> Settings:
    """Get singleton Settings instance.

    Uses @lru_cache to ensure only one instance is created.
    This follows the singleton pattern from CODING_PATTERNS_ANALYSIS.md.

    Returns:
        Cached Settings instance.
    """
    return Settings()
