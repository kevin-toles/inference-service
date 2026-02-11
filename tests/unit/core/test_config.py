"""Tests for core configuration module.

TDD RED Phase: These tests define the expected behavior of Settings class.
Reference: WBS-INF2 Exit Criteria, CODING_PATTERNS_ANALYSIS.md

Tests verify:
- AC-2.1: Pydantic Settings loads all environment variables
- AC-2.4: Configuration validates on startup with clear error messages
"""

import os
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest
from pydantic import ValidationError

if TYPE_CHECKING:
    from src.core.config import Settings

# =============================================================================
# Test Constants
# =============================================================================

# Environment variables that skip path validation for unit tests
# Production containers will have real paths mounted
SKIP_VALIDATION_ENV = {"INFERENCE_SKIP_PATH_VALIDATION": "true"}


class TestSettingsDefaults:
    """Test Settings class default values."""

    def test_default_port_is_8085(self) -> None:
        """Port defaults to 8085 per ARCHITECTURE.md."""
        from src.core.config import Settings

        with patch.dict(os.environ, SKIP_VALIDATION_ENV, clear=True):
            settings = Settings()
        assert settings.port == 8085

    def test_default_host_is_localhost(self) -> None:
        """Host defaults to :: for dual-stack IPv4+IPv6 (C-7 fix)."""
        from src.core.config import Settings

        with patch.dict(os.environ, SKIP_VALIDATION_ENV, clear=True):
            settings = Settings()
        assert settings.host == "::"

    def test_default_log_level_is_info(self) -> None:
        """Log level defaults to INFO."""
        from src.core.config import Settings

        with patch.dict(os.environ, SKIP_VALIDATION_ENV, clear=True):
            settings = Settings()
        assert settings.log_level == "INFO"

    def test_default_environment_is_development(self) -> None:
        """Environment defaults to development."""
        from src.core.config import Settings

        with patch.dict(os.environ, SKIP_VALIDATION_ENV, clear=True):
            settings = Settings()
        assert settings.environment == "development"

    def test_default_service_name(self) -> None:
        """Service name defaults to inference-service."""
        from src.core.config import Settings

        with patch.dict(os.environ, SKIP_VALIDATION_ENV, clear=True):
            settings = Settings()
        assert settings.service_name == "inference-service"

    def test_default_models_dir_is_container_path(self) -> None:
        """models_dir defaults to /app/models (container path from constants)."""
        from src.core.config import Settings
        from src.core.constants import CONTAINER_MODELS_DIR

        with patch.dict(os.environ, SKIP_VALIDATION_ENV, clear=True):
            settings = Settings()
        assert settings.models_dir == CONTAINER_MODELS_DIR
        assert settings.models_dir == "/app/models"

    def test_default_config_dir_is_container_path(self) -> None:
        """config_dir defaults to /app/config (container path from constants)."""
        from src.core.config import Settings
        from src.core.constants import CONTAINER_CONFIG_DIR

        with patch.dict(os.environ, SKIP_VALIDATION_ENV, clear=True):
            settings = Settings()
        assert settings.config_dir == CONTAINER_CONFIG_DIR
        assert settings.config_dir == "/app/config"


class TestSettingsEnvironmentVariables:
    """Test Settings loads from INFERENCE_* prefixed environment variables."""

    def test_loads_port_from_env(self) -> None:
        """Settings loads INFERENCE_PORT from environment.
        
        Exit Criteria: INFERENCE_PORT=9999 → settings.port == 9999
        """
        from src.core.config import Settings

        env = {**SKIP_VALIDATION_ENV, "INFERENCE_PORT": "9999"}
        with patch.dict(os.environ, env, clear=True):
            settings = Settings()
        assert settings.port == 9999

    def test_loads_host_from_env(self) -> None:
        """Settings loads INFERENCE_HOST from environment."""
        from src.core.config import Settings

        env = {**SKIP_VALIDATION_ENV, "INFERENCE_HOST": "127.0.0.1"}
        with patch.dict(os.environ, env, clear=True):
            settings = Settings()
        assert settings.host == "127.0.0.1"

    def test_loads_log_level_from_env(self) -> None:
        """Settings loads INFERENCE_LOG_LEVEL from environment."""
        from src.core.config import Settings

        env = {**SKIP_VALIDATION_ENV, "INFERENCE_LOG_LEVEL": "DEBUG"}
        with patch.dict(os.environ, env, clear=True):
            settings = Settings()
        assert settings.log_level == "DEBUG"

    def test_loads_environment_from_env(self) -> None:
        """Settings loads INFERENCE_ENVIRONMENT from environment."""
        from src.core.config import Settings

        env = {**SKIP_VALIDATION_ENV, "INFERENCE_ENVIRONMENT": "production"}
        with patch.dict(os.environ, env, clear=True):
            settings = Settings()
        assert settings.environment == "production"

    def test_env_prefix_is_inference(self) -> None:
        """All env vars must be prefixed with INFERENCE_."""
        from src.core.config import Settings

        # Without prefix, should use default
        env = {**SKIP_VALIDATION_ENV, "PORT": "1234"}
        with patch.dict(os.environ, env, clear=True):
            settings = Settings()
        assert settings.port == 8085  # Default, not 1234


class TestSettingsValidation:
    """Test Settings validation rules."""

    def test_port_must_be_in_valid_range(self) -> None:
        """Port must be between 1 and 65535.
        
        Exit Criteria: Invalid config raises ValidationError with clear message.
        """
        from src.core.config import Settings

        env = {**SKIP_VALIDATION_ENV, "INFERENCE_PORT": "0"}
        with patch.dict(os.environ, env, clear=True):
            with pytest.raises(ValidationError) as exc_info:
                Settings()
        assert "port" in str(exc_info.value).lower()

    def test_port_rejects_negative(self) -> None:
        """Port rejects negative values."""
        from src.core.config import Settings

        env = {**SKIP_VALIDATION_ENV, "INFERENCE_PORT": "-1"}
        with patch.dict(os.environ, env, clear=True):
            with pytest.raises(ValidationError):
                Settings()

    def test_port_rejects_too_high(self) -> None:
        """Port rejects values over 65535."""
        from src.core.config import Settings

        env = {**SKIP_VALIDATION_ENV, "INFERENCE_PORT": "70000"}
        with patch.dict(os.environ, env, clear=True):
            with pytest.raises(ValidationError):
                Settings()

    def test_environment_must_be_valid_literal(self) -> None:
        """Environment must be development, staging, or production."""
        from src.core.config import Settings

        env = {**SKIP_VALIDATION_ENV, "INFERENCE_ENVIRONMENT": "invalid"}
        with patch.dict(os.environ, env, clear=True):
            with pytest.raises(ValidationError) as exc_info:
                Settings()
        assert "environment" in str(exc_info.value).lower()

    def test_log_level_validates_known_levels(self) -> None:
        """Log level must be valid (DEBUG, INFO, WARNING, ERROR, CRITICAL)."""
        from src.core.config import Settings

        env = {**SKIP_VALIDATION_ENV, "INFERENCE_LOG_LEVEL": "INVALID"}
        with patch.dict(os.environ, env, clear=True):
            with pytest.raises(ValidationError) as exc_info:
                Settings()
        assert "log_level" in str(exc_info.value).lower()

    def test_log_level_normalizes_to_uppercase(self) -> None:
        """Log level is normalized to uppercase."""
        from src.core.config import Settings

        env = {**SKIP_VALIDATION_ENV, "INFERENCE_LOG_LEVEL": "debug"}
        with patch.dict(os.environ, env, clear=True):
            settings = Settings()
        assert settings.log_level == "DEBUG"


class TestSettingsPathValidation:
    """Test path existence validation."""

    def test_validates_models_dir_exists(self, tmp_path) -> None:
        """Raises error if models_dir does not exist."""
        from src.core.config import Settings

        fake_path = "/nonexistent/models/path"
        env = {"INFERENCE_MODELS_DIR": fake_path, "INFERENCE_CONFIG_DIR": str(tmp_path)}
        with patch.dict(os.environ, env, clear=True):
            with pytest.raises(ValidationError) as exc_info:
                Settings()
        assert "models_dir" in str(exc_info.value)
        assert fake_path in str(exc_info.value)

    def test_validates_config_dir_exists(self, tmp_path) -> None:
        """Raises error if config_dir does not exist."""
        from src.core.config import Settings

        fake_path = "/nonexistent/config/path"
        env = {"INFERENCE_MODELS_DIR": str(tmp_path), "INFERENCE_CONFIG_DIR": fake_path}
        with patch.dict(os.environ, env, clear=True):
            with pytest.raises(ValidationError) as exc_info:
                Settings()
        assert "config_dir" in str(exc_info.value)
        assert fake_path in str(exc_info.value)

    def test_validation_can_be_skipped(self) -> None:
        """Path validation can be skipped for testing."""
        from src.core.config import Settings

        env = {
            "INFERENCE_MODELS_DIR": "/fake/path",
            "INFERENCE_CONFIG_DIR": "/fake/config",
            "INFERENCE_SKIP_PATH_VALIDATION": "true",
        }
        with patch.dict(os.environ, env, clear=True):
            settings = Settings()
        # Should not raise, paths can be fake when validation skipped
        assert settings.models_dir == "/fake/path"

    def test_passes_when_paths_exist(self, tmp_path) -> None:
        """Validation passes when paths exist."""
        from src.core.config import Settings

        models_dir = tmp_path / "models"
        config_dir = tmp_path / "config"
        models_dir.mkdir()
        config_dir.mkdir()

        env = {
            "INFERENCE_MODELS_DIR": str(models_dir),
            "INFERENCE_CONFIG_DIR": str(config_dir),
        }
        with patch.dict(os.environ, env, clear=True):
            settings = Settings()
        assert settings.models_dir == str(models_dir)
        assert settings.config_dir == str(config_dir)


class TestSettingsModelConfiguration:
    """Test model and inference-specific settings."""

    def test_models_dir_default(self) -> None:
        """Models directory has sensible default."""
        from src.core.config import Settings

        with patch.dict(os.environ, SKIP_VALIDATION_ENV, clear=True):
            settings = Settings()
        assert settings.models_dir is not None

    def test_loads_models_dir_from_env(self) -> None:
        """Settings loads INFERENCE_MODELS_DIR from environment."""
        from src.core.config import Settings

        custom_path = "/custom/models/path"
        env = {**SKIP_VALIDATION_ENV, "INFERENCE_MODELS_DIR": custom_path}
        with patch.dict(os.environ, env, clear=True):
            settings = Settings()
        assert settings.models_dir == custom_path

    def test_default_gpu_layers(self) -> None:
        """GPU layers defaults to -1 (all on GPU)."""
        from src.core.config import Settings

        with patch.dict(os.environ, SKIP_VALIDATION_ENV, clear=True):
            settings = Settings()
        assert settings.gpu_layers == -1

    def test_default_gpu_index(self) -> None:
        """GPU index defaults to 0 (first GPU).
        
        WBS-GPU6: Add INFERENCE_GPU_INDEX env var for multi-GPU.
        """
        from src.core.config import Settings

        with patch.dict(os.environ, SKIP_VALIDATION_ENV, clear=True):
            settings = Settings()
        assert settings.gpu_index == 0

    def test_loads_gpu_index_from_env(self) -> None:
        """Settings loads INFERENCE_GPU_INDEX from environment.
        
        WBS-GPU6: Multi-GPU selection via environment variable.
        """
        from src.core.config import Settings

        env = {**SKIP_VALIDATION_ENV, "INFERENCE_GPU_INDEX": "1"}
        with patch.dict(os.environ, env, clear=True):
            settings = Settings()
        assert settings.gpu_index == 1

    def test_gpu_index_rejects_negative(self) -> None:
        """GPU index must be >= 0.
        
        WBS-GPU7: Validate GPU index is non-negative.
        """
        from src.core.config import Settings

        env = {**SKIP_VALIDATION_ENV, "INFERENCE_GPU_INDEX": "-1"}
        with patch.dict(os.environ, env, clear=True):
            with pytest.raises(ValidationError) as exc_info:
                Settings()
        assert "gpu_index" in str(exc_info.value).lower()

    def test_default_backend(self) -> None:
        """Backend defaults to llamacpp."""
        from src.core.config import Settings

        with patch.dict(os.environ, SKIP_VALIDATION_ENV, clear=True):
            settings = Settings()
        assert settings.backend == "llamacpp"


class TestSettingsOrchestration:
    """Test orchestration-related settings."""

    def test_default_orchestration_mode(self) -> None:
        """Orchestration mode defaults to single."""
        from src.core.config import Settings

        with patch.dict(os.environ, SKIP_VALIDATION_ENV, clear=True):
            settings = Settings()
        assert settings.orchestration_mode == "single"

    def test_orchestration_mode_validates(self) -> None:
        """Orchestration mode must be valid Literal value."""
        from src.core.config import Settings

        env = {**SKIP_VALIDATION_ENV, "INFERENCE_ORCHESTRATION_MODE": "invalid_mode"}
        with patch.dict(os.environ, env, clear=True):
            with pytest.raises(ValidationError) as exc_info:
                Settings()
        assert "orchestration_mode" in str(exc_info.value).lower()
