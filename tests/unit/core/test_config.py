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


class TestSettingsDefaults:
    """Test Settings class default values."""

    def test_default_port_is_8085(self) -> None:
        """Port defaults to 8085 per ARCHITECTURE.md."""
        from src.core.config import Settings

        with patch.dict(os.environ, {}, clear=True):
            settings = Settings()
        assert settings.port == 8085

    def test_default_host_is_localhost(self) -> None:
        """Host defaults to 0.0.0.0 for container deployment."""
        from src.core.config import Settings

        with patch.dict(os.environ, {}, clear=True):
            settings = Settings()
        assert settings.host == "0.0.0.0"

    def test_default_log_level_is_info(self) -> None:
        """Log level defaults to INFO."""
        from src.core.config import Settings

        with patch.dict(os.environ, {}, clear=True):
            settings = Settings()
        assert settings.log_level == "INFO"

    def test_default_environment_is_development(self) -> None:
        """Environment defaults to development."""
        from src.core.config import Settings

        with patch.dict(os.environ, {}, clear=True):
            settings = Settings()
        assert settings.environment == "development"

    def test_default_service_name(self) -> None:
        """Service name defaults to inference-service."""
        from src.core.config import Settings

        with patch.dict(os.environ, {}, clear=True):
            settings = Settings()
        assert settings.service_name == "inference-service"


class TestSettingsEnvironmentVariables:
    """Test Settings loads from INFERENCE_* prefixed environment variables."""

    def test_loads_port_from_env(self) -> None:
        """Settings loads INFERENCE_PORT from environment.
        
        Exit Criteria: INFERENCE_PORT=9999 → settings.port == 9999
        """
        from src.core.config import Settings

        with patch.dict(os.environ, {"INFERENCE_PORT": "9999"}, clear=True):
            settings = Settings()
        assert settings.port == 9999

    def test_loads_host_from_env(self) -> None:
        """Settings loads INFERENCE_HOST from environment."""
        from src.core.config import Settings

        with patch.dict(os.environ, {"INFERENCE_HOST": "127.0.0.1"}, clear=True):
            settings = Settings()
        assert settings.host == "127.0.0.1"

    def test_loads_log_level_from_env(self) -> None:
        """Settings loads INFERENCE_LOG_LEVEL from environment."""
        from src.core.config import Settings

        with patch.dict(os.environ, {"INFERENCE_LOG_LEVEL": "DEBUG"}, clear=True):
            settings = Settings()
        assert settings.log_level == "DEBUG"

    def test_loads_environment_from_env(self) -> None:
        """Settings loads INFERENCE_ENVIRONMENT from environment."""
        from src.core.config import Settings

        with patch.dict(os.environ, {"INFERENCE_ENVIRONMENT": "production"}, clear=True):
            settings = Settings()
        assert settings.environment == "production"

    def test_env_prefix_is_inference(self) -> None:
        """All env vars must be prefixed with INFERENCE_."""
        from src.core.config import Settings

        # Without prefix, should use default
        with patch.dict(os.environ, {"PORT": "1234"}, clear=True):
            settings = Settings()
        assert settings.port == 8085  # Default, not 1234


class TestSettingsValidation:
    """Test Settings validation rules."""

    def test_port_must_be_in_valid_range(self) -> None:
        """Port must be between 1 and 65535.
        
        Exit Criteria: Invalid config raises ValidationError with clear message.
        """
        from src.core.config import Settings

        with patch.dict(os.environ, {"INFERENCE_PORT": "0"}, clear=True):
            with pytest.raises(ValidationError) as exc_info:
                Settings()
        assert "port" in str(exc_info.value).lower()

    def test_port_rejects_negative(self) -> None:
        """Port rejects negative values."""
        from src.core.config import Settings

        with patch.dict(os.environ, {"INFERENCE_PORT": "-1"}, clear=True):
            with pytest.raises(ValidationError):
                Settings()

    def test_port_rejects_too_high(self) -> None:
        """Port rejects values over 65535."""
        from src.core.config import Settings

        with patch.dict(os.environ, {"INFERENCE_PORT": "70000"}, clear=True):
            with pytest.raises(ValidationError):
                Settings()

    def test_environment_must_be_valid_literal(self) -> None:
        """Environment must be development, staging, or production."""
        from src.core.config import Settings

        with patch.dict(os.environ, {"INFERENCE_ENVIRONMENT": "invalid"}, clear=True):
            with pytest.raises(ValidationError) as exc_info:
                Settings()
        assert "environment" in str(exc_info.value).lower()

    def test_log_level_validates_known_levels(self) -> None:
        """Log level must be valid (DEBUG, INFO, WARNING, ERROR, CRITICAL)."""
        from src.core.config import Settings

        with patch.dict(os.environ, {"INFERENCE_LOG_LEVEL": "INVALID"}, clear=True):
            with pytest.raises(ValidationError) as exc_info:
                Settings()
        assert "log_level" in str(exc_info.value).lower()

    def test_log_level_normalizes_to_uppercase(self) -> None:
        """Log level is normalized to uppercase."""
        from src.core.config import Settings

        with patch.dict(os.environ, {"INFERENCE_LOG_LEVEL": "debug"}, clear=True):
            settings = Settings()
        assert settings.log_level == "DEBUG"


class TestSettingsModelConfiguration:
    """Test model and inference-specific settings."""

    def test_models_dir_default(self) -> None:
        """Models directory has sensible default."""
        from src.core.config import Settings

        with patch.dict(os.environ, {}, clear=True):
            settings = Settings()
        assert settings.models_dir is not None

    def test_loads_models_dir_from_env(self) -> None:
        """Settings loads INFERENCE_MODELS_DIR from environment."""
        from src.core.config import Settings

        custom_path = "/custom/models/path"
        with patch.dict(os.environ, {"INFERENCE_MODELS_DIR": custom_path}, clear=True):
            settings = Settings()
        assert settings.models_dir == custom_path

    def test_default_gpu_layers(self) -> None:
        """GPU layers defaults to -1 (all on GPU)."""
        from src.core.config import Settings

        with patch.dict(os.environ, {}, clear=True):
            settings = Settings()
        assert settings.gpu_layers == -1

    def test_default_backend(self) -> None:
        """Backend defaults to llamacpp."""
        from src.core.config import Settings

        with patch.dict(os.environ, {}, clear=True):
            settings = Settings()
        assert settings.backend == "llamacpp"


class TestSettingsOrchestration:
    """Test orchestration-related settings."""

    def test_default_orchestration_mode(self) -> None:
        """Orchestration mode defaults to single."""
        from src.core.config import Settings

        with patch.dict(os.environ, {}, clear=True):
            settings = Settings()
        assert settings.orchestration_mode == "single"

    def test_orchestration_mode_validates(self) -> None:
        """Orchestration mode must be valid."""
        from src.core.config import Settings

        with patch.dict(
            os.environ, {"INFERENCE_ORCHESTRATION_MODE": "invalid_mode"}, clear=True
        ):
            with pytest.raises(ValidationError):
                Settings()


class TestSettingsSingleton:
    """Test get_settings() singleton pattern."""

    def test_get_settings_returns_settings_instance(self) -> None:
        """get_settings() returns a Settings instance."""
        from src.core.config import get_settings

        settings = get_settings()
        assert settings is not None
        assert hasattr(settings, "port")
        assert hasattr(settings, "log_level")

    def test_get_settings_is_cached(self) -> None:
        """get_settings() returns same instance (lru_cache singleton)."""
        from src.core.config import get_settings

        settings1 = get_settings()
        settings2 = get_settings()
        assert settings1 is settings2
