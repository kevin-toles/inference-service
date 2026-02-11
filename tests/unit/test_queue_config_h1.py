"""Tests for H-1 Task 2.3: Queue configuration via environment variables.

Verifies:
- AC: All queue parameters configurable via environment
- AC: Defaults are safe: max_concurrent=1, max_size=10, strategy=fifo
- AC: QueueManager receives settings-driven values in lifespan
"""

import os
from unittest.mock import patch

import pytest
from pydantic import ValidationError

# Environment variables that skip path validation for unit tests
SKIP_VALIDATION_ENV = {"INFERENCE_SKIP_PATH_VALIDATION": "true"}


# =============================================================================
# Settings Defaults
# =============================================================================


class TestQueueSettingsDefaults:
    """Verify safe defaults for all queue-related settings."""

    def test_default_max_concurrent_is_1(self) -> None:
        """max_concurrent defaults to 1 (single GPU serialization)."""
        from src.core.config import Settings

        with patch.dict(os.environ, SKIP_VALIDATION_ENV, clear=True):
            settings = Settings()
        assert settings.max_concurrent == 1

    def test_default_queue_max_size_is_10(self) -> None:
        """queue_max_size defaults to 10."""
        from src.core.config import Settings

        with patch.dict(os.environ, SKIP_VALIDATION_ENV, clear=True):
            settings = Settings()
        assert settings.queue_max_size == 10

    def test_default_queue_strategy_is_fifo(self) -> None:
        """queue_strategy defaults to 'fifo'."""
        from src.core.config import Settings

        with patch.dict(os.environ, SKIP_VALIDATION_ENV, clear=True):
            settings = Settings()
        assert settings.queue_strategy == "fifo"


# =============================================================================
# Settings from Environment Variables
# =============================================================================


class TestQueueSettingsFromEnv:
    """Verify settings load from INFERENCE_* environment variables."""

    def test_max_concurrent_from_env(self) -> None:
        """INFERENCE_MAX_CONCURRENT overrides default."""
        from src.core.config import Settings

        env = {**SKIP_VALIDATION_ENV, "INFERENCE_MAX_CONCURRENT": "4"}
        with patch.dict(os.environ, env, clear=True):
            settings = Settings()
        assert settings.max_concurrent == 4

    def test_queue_max_size_from_env(self) -> None:
        """INFERENCE_QUEUE_MAX_SIZE overrides default."""
        from src.core.config import Settings

        env = {**SKIP_VALIDATION_ENV, "INFERENCE_QUEUE_MAX_SIZE": "50"}
        with patch.dict(os.environ, env, clear=True):
            settings = Settings()
        assert settings.queue_max_size == 50

    def test_queue_strategy_from_env_fifo(self) -> None:
        """INFERENCE_QUEUE_STRATEGY=fifo accepted."""
        from src.core.config import Settings

        env = {**SKIP_VALIDATION_ENV, "INFERENCE_QUEUE_STRATEGY": "fifo"}
        with patch.dict(os.environ, env, clear=True):
            settings = Settings()
        assert settings.queue_strategy == "fifo"

    def test_queue_strategy_from_env_priority(self) -> None:
        """INFERENCE_QUEUE_STRATEGY=priority accepted."""
        from src.core.config import Settings

        env = {**SKIP_VALIDATION_ENV, "INFERENCE_QUEUE_STRATEGY": "priority"}
        with patch.dict(os.environ, env, clear=True):
            settings = Settings()
        assert settings.queue_strategy == "priority"


# =============================================================================
# Settings Validation
# =============================================================================


class TestQueueSettingsValidation:
    """Verify Pydantic validation rejects invalid values."""

    def test_max_concurrent_zero_rejected(self) -> None:
        """max_concurrent=0 violates ge=1 constraint."""
        from src.core.config import Settings

        env = {**SKIP_VALIDATION_ENV, "INFERENCE_MAX_CONCURRENT": "0"}
        with patch.dict(os.environ, env, clear=True):
            with pytest.raises(ValidationError):
                Settings()

    def test_max_concurrent_negative_rejected(self) -> None:
        """Negative max_concurrent rejected."""
        from src.core.config import Settings

        env = {**SKIP_VALIDATION_ENV, "INFERENCE_MAX_CONCURRENT": "-1"}
        with patch.dict(os.environ, env, clear=True):
            with pytest.raises(ValidationError):
                Settings()

    def test_queue_max_size_zero_rejected(self) -> None:
        """queue_max_size=0 violates ge=1 constraint."""
        from src.core.config import Settings

        env = {**SKIP_VALIDATION_ENV, "INFERENCE_QUEUE_MAX_SIZE": "0"}
        with patch.dict(os.environ, env, clear=True):
            with pytest.raises(ValidationError):
                Settings()

    def test_queue_max_size_over_1000_rejected(self) -> None:
        """queue_max_size=1001 violates le=1000 constraint."""
        from src.core.config import Settings

        env = {**SKIP_VALIDATION_ENV, "INFERENCE_QUEUE_MAX_SIZE": "1001"}
        with patch.dict(os.environ, env, clear=True):
            with pytest.raises(ValidationError):
                Settings()

    def test_queue_strategy_invalid_rejected(self) -> None:
        """Invalid strategy string rejected."""
        from src.core.config import Settings

        env = {**SKIP_VALIDATION_ENV, "INFERENCE_QUEUE_STRATEGY": "random"}
        with patch.dict(os.environ, env, clear=True):
            with pytest.raises(ValidationError):
                Settings()


# =============================================================================
# QueueManager max_size wiring
# =============================================================================


class TestQueueManagerMaxSize:
    """Verify QueueManager respects max_size parameter."""

    def test_default_max_size_is_unbounded(self) -> None:
        """QueueManager without max_size has unbounded queue."""
        from src.services.queue_manager import QueueManager

        qm = QueueManager(max_concurrent=1)
        assert qm.max_size == 0

    def test_max_size_stored(self) -> None:
        """max_size value is accessible via property."""
        from src.services.queue_manager import QueueManager

        qm = QueueManager(max_concurrent=1, max_size=10)
        assert qm.max_size == 10

    def test_max_size_applied_to_fifo_queue(self) -> None:
        """FIFO queue respects maxsize."""
        from src.services.queue_manager import QueueManager, QueueStrategy

        qm = QueueManager(max_concurrent=1, strategy=QueueStrategy.FIFO, max_size=5)
        assert qm._queue.maxsize == 5

    def test_max_size_applied_to_priority_queue(self) -> None:
        """Priority queue respects maxsize."""
        from src.services.queue_manager import QueueManager, QueueStrategy

        qm = QueueManager(
            max_concurrent=1, strategy=QueueStrategy.PRIORITY, max_size=5
        )
        assert qm._queue.maxsize == 5


# =============================================================================
# Lifespan wiring with env vars
# =============================================================================


class TestLifespanQueueConfig:
    """Verify lifespan reads queue config from Settings."""

    def test_lifespan_uses_settings_max_concurrent(self) -> None:
        """QueueManager.max_concurrent matches INFERENCE_MAX_CONCURRENT."""
        from starlette.testclient import TestClient

        from src.main import app

        env = {
            "INFERENCE_SKIP_PATH_VALIDATION": "true",
            "INFERENCE_MODELS_DIR": "/Users/kevintoles/POC/ai-models/models",
            "INFERENCE_CONFIG_DIR": "/Users/kevintoles/POC/inference-service/config",
            "INFERENCE_MAX_CONCURRENT": "3",
            "INFERENCE_QUEUE_MAX_SIZE": "25",
            "INFERENCE_QUEUE_STRATEGY": "priority",
        }
        from src.core.config import get_settings

        get_settings.cache_clear()
        try:
            with patch.dict(os.environ, env):
                get_settings.cache_clear()
                with TestClient(app) as client:
                    qm = app.state.queue_manager
                    assert qm.max_concurrent == 3
                    assert qm.max_size == 25
                    assert qm._strategy.value == "priority"
        finally:
            get_settings.cache_clear()

    def test_lifespan_defaults_safe(self) -> None:
        """Without queue env vars, defaults are safe (1, 10, fifo)."""
        from starlette.testclient import TestClient

        from src.main import app

        # Only set required env vars, no queue overrides
        env = {
            "INFERENCE_SKIP_PATH_VALIDATION": "true",
            "INFERENCE_MODELS_DIR": "/Users/kevintoles/POC/ai-models/models",
            "INFERENCE_CONFIG_DIR": "/Users/kevintoles/POC/inference-service/config",
        }
        # Build a clean environment without any INFERENCE_MAX_CONCURRENT etc.
        clean_env = {
            k: v
            for k, v in os.environ.items()
            if not k.startswith("INFERENCE_")
        }
        clean_env.update(env)
        from src.core.config import get_settings

        get_settings.cache_clear()
        try:
            with patch.dict(os.environ, clean_env, clear=True):
                get_settings.cache_clear()
                with TestClient(app) as client:
                    qm = app.state.queue_manager
                    assert qm.max_concurrent == 1
                    assert qm.max_size == 10
                    assert qm._strategy.value == "fifo"
        finally:
            get_settings.cache_clear()
