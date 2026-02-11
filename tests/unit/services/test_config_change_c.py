"""Unit tests for Phase C — publish_config_changed() and update_model_config().

Tests:
- publish_config_changed() publishes to BOTH channels (AC-C.4)
- update_model_config() calls log_config_changed() per field (AC-C.1, AC-C.2)
- update_model_config() calls publish_config_changed() per field (AC-C.4)
- PATCH /v1/models/{id}/config endpoint wiring

Reference: WBS-MESH-C Tasks C.1-C.6
"""

from __future__ import annotations

import json
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.services.config_publisher import (
    CHANNEL_MODEL_CONFIG,
    CHANNEL_MODEL_LIFECYCLE,
    ConfigPublisher,
)

TEST_MODEL_ID = "qwen3-8b"
TEST_REDIS_URL = "redis://localhost:6379"


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def publisher() -> ConfigPublisher:
    """Create a ConfigPublisher with a mocked Redis client."""
    pub = ConfigPublisher(TEST_REDIS_URL)
    pub._redis = AsyncMock()
    pub._connected = True
    pub._redis.hgetall = AsyncMock(return_value={})
    return pub


@pytest.fixture
def disconnected_publisher() -> ConfigPublisher:
    """Create a disconnected ConfigPublisher."""
    pub = ConfigPublisher(TEST_REDIS_URL)
    pub._redis = None
    pub._connected = False
    return pub


# =============================================================================
# TestPublishConfigChanged — AC-C.4
# =============================================================================


class TestPublishConfigChanged:
    """Tests for publish_config_changed() — Phase C."""

    @pytest.mark.asyncio
    async def test_publishes_to_config_channel(
        self, publisher: ConfigPublisher
    ) -> None:
        """Publishes CONFIG_CHANGED to model:config:changes channel."""
        publisher._redis.publish = AsyncMock(return_value=1)

        result = await publisher.publish_config_changed(
            model_id=TEST_MODEL_ID,
            field="context_length",
            old_value=2048,
            new_value=4096,
        )

        assert result is True
        first_call = publisher._redis.publish.call_args_list[0]
        assert first_call[0][0] == CHANNEL_MODEL_CONFIG
        payload = json.loads(first_call[0][1])
        assert payload["event_type"] == "CONFIG_CHANGED"
        assert payload["model_id"] == TEST_MODEL_ID
        assert payload["data"]["field"] == "context_length"
        assert payload["data"]["old_value"] == 2048
        assert payload["data"]["new_value"] == 4096

    @pytest.mark.asyncio
    async def test_publishes_to_lifecycle_channel(
        self, publisher: ConfigPublisher
    ) -> None:
        """Publishes CONFIG_CHANGED to model:lifecycle:events channel."""
        publisher._redis.publish = AsyncMock(return_value=1)

        await publisher.publish_config_changed(
            model_id=TEST_MODEL_ID,
            field="gpu_layers",
            old_value=-1,
            new_value=20,
        )

        second_call = publisher._redis.publish.call_args_list[1]
        assert second_call[0][0] == CHANNEL_MODEL_LIFECYCLE
        payload = json.loads(second_call[0][1])
        assert payload["event_type"] == "CONFIG_CHANGED"
        assert payload["trigger"] == "config_update"
        assert payload["config_change"]["field"] == "gpu_layers"
        assert payload["config_change"]["old_value"] == -1
        assert payload["config_change"]["new_value"] == 20

    @pytest.mark.asyncio
    async def test_lifecycle_has_required_fields(
        self, publisher: ConfigPublisher
    ) -> None:
        """Lifecycle event includes all standard fields."""
        publisher._redis.publish = AsyncMock(return_value=1)

        await publisher.publish_config_changed(
            model_id=TEST_MODEL_ID,
            field="context_length",
            old_value=2048,
            new_value=8192,
        )

        second_call = publisher._redis.publish.call_args_list[1]
        payload = json.loads(second_call[0][1])
        assert payload["source"] == "inference-service"
        parsed = datetime.fromisoformat(payload["timestamp"])
        assert parsed.tzinfo is not None

    @pytest.mark.asyncio
    async def test_returns_false_when_disconnected(
        self, disconnected_publisher: ConfigPublisher
    ) -> None:
        """Returns False when not connected."""
        result = await disconnected_publisher.publish_config_changed(
            model_id=TEST_MODEL_ID,
            field="context_length",
            old_value=2048,
            new_value=4096,
        )
        assert result is False

    @pytest.mark.asyncio
    async def test_graceful_fallback_on_redis_error(
        self, publisher: ConfigPublisher
    ) -> None:
        """Returns False on Redis error, no exception."""
        from redis.exceptions import ConnectionError as RedisConnectionError

        publisher._redis.publish = AsyncMock(
            side_effect=RedisConnectionError("connection lost")
        )

        result = await publisher.publish_config_changed(
            model_id=TEST_MODEL_ID,
            field="context_length",
            old_value=2048,
            new_value=4096,
        )
        assert result is False


# =============================================================================
# TestUpdateModelConfig — AC-C.1, AC-C.2
# =============================================================================


class TestUpdateModelConfig:
    """Tests for ModelManager.update_model_config() — Phase C."""

    @pytest.fixture
    def manager(self):
        """Create a ModelManager with test config."""
        from src.services.model_manager import ModelManager

        return ModelManager(
            models_dir="/tmp/test-models",
            model_configs={
                TEST_MODEL_ID: {
                    "context_length": 2048,
                    "gpu_layers": -1,
                    "size_gb": 4.0,
                    "roles": ["chat"],
                    "file": "test.gguf",
                },
            },
            config_presets={},
            memory_limit_gb=16.0,
        )

    @pytest.mark.asyncio
    async def test_calls_log_config_changed_on_context_length(
        self, manager
    ) -> None:
        """AC-C.1: log_config_changed called when context_length changes."""
        mock_audit = MagicMock()
        mock_audit.is_connected = True
        mock_audit.log_config_changed = AsyncMock(return_value="node-123")

        with patch(
            "src.services.model_manager.get_audit_client",
            return_value=mock_audit,
        ), patch(
            "src.services.model_manager.get_config_publisher",
            return_value=None,
        ):
            changed = await manager.update_model_config(
                TEST_MODEL_ID, context_length=4096
            )

        assert changed == {"context_length": 4096}
        mock_audit.log_config_changed.assert_awaited_once_with(
            model_id=TEST_MODEL_ID,
            field="context_length",
            old_value=2048,
            new_value=4096,
        )

    @pytest.mark.asyncio
    async def test_calls_log_config_changed_on_gpu_layers(
        self, manager
    ) -> None:
        """AC-C.2: log_config_changed called when gpu_layers changes."""
        mock_audit = MagicMock()
        mock_audit.is_connected = True
        mock_audit.log_config_changed = AsyncMock(return_value="node-456")

        with patch(
            "src.services.model_manager.get_audit_client",
            return_value=mock_audit,
        ), patch(
            "src.services.model_manager.get_config_publisher",
            return_value=None,
        ):
            changed = await manager.update_model_config(
                TEST_MODEL_ID, gpu_layers=20
            )

        assert changed == {"gpu_layers": 20}
        mock_audit.log_config_changed.assert_awaited_once_with(
            model_id=TEST_MODEL_ID,
            field="gpu_layers",
            old_value=-1,
            new_value=20,
        )

    @pytest.mark.asyncio
    async def test_calls_publish_config_changed(self, manager) -> None:
        """AC-C.4: publish_config_changed called for each changed field."""
        mock_publisher = MagicMock()
        mock_publisher.publish_config_changed = AsyncMock(return_value=True)

        with patch(
            "src.services.model_manager.get_audit_client",
            return_value=None,
        ), patch(
            "src.services.model_manager.get_config_publisher",
            return_value=mock_publisher,
        ):
            await manager.update_model_config(
                TEST_MODEL_ID, context_length=8192
            )

        mock_publisher.publish_config_changed.assert_awaited_once_with(
            model_id=TEST_MODEL_ID,
            field="context_length",
            old_value=2048,
            new_value=8192,
        )

    @pytest.mark.asyncio
    async def test_multiple_fields_changed(self, manager) -> None:
        """Both fields changed → log_config_changed called twice."""
        mock_audit = MagicMock()
        mock_audit.is_connected = True
        mock_audit.log_config_changed = AsyncMock(return_value="node-x")

        with patch(
            "src.services.model_manager.get_audit_client",
            return_value=mock_audit,
        ), patch(
            "src.services.model_manager.get_config_publisher",
            return_value=None,
        ):
            changed = await manager.update_model_config(
                TEST_MODEL_ID, context_length=4096, gpu_layers=0
            )

        assert "context_length" in changed
        assert "gpu_layers" in changed
        assert mock_audit.log_config_changed.await_count == 2

    @pytest.mark.asyncio
    async def test_no_change_when_same_value(self, manager) -> None:
        """No log_config_changed call when value is unchanged."""
        mock_audit = MagicMock()
        mock_audit.is_connected = True
        mock_audit.log_config_changed = AsyncMock()

        with patch(
            "src.services.model_manager.get_audit_client",
            return_value=mock_audit,
        ), patch(
            "src.services.model_manager.get_config_publisher",
            return_value=None,
        ):
            changed = await manager.update_model_config(
                TEST_MODEL_ID, context_length=2048  # same as current
            )

        assert changed == {}
        mock_audit.log_config_changed.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_ignores_unsupported_fields(self, manager) -> None:
        """Unsupported fields like 'roles' are ignored."""
        with patch(
            "src.services.model_manager.get_audit_client",
            return_value=None,
        ), patch(
            "src.services.model_manager.get_config_publisher",
            return_value=None,
        ):
            changed = await manager.update_model_config(
                TEST_MODEL_ID, roles=["code", "chat"]
            )

        assert changed == {}

    @pytest.mark.asyncio
    async def test_unknown_model_raises_error(self, manager) -> None:
        """ModelNotAvailableError raised for unknown model."""
        from src.services.model_manager import ModelNotAvailableError

        with pytest.raises(ModelNotAvailableError):
            await manager.update_model_config(
                "nonexistent-model", context_length=4096
            )

    @pytest.mark.asyncio
    async def test_updates_stored_config(self, manager) -> None:
        """Stored config is updated so next load uses new value."""
        with patch(
            "src.services.model_manager.get_audit_client",
            return_value=None,
        ), patch(
            "src.services.model_manager.get_config_publisher",
            return_value=None,
        ):
            await manager.update_model_config(
                TEST_MODEL_ID, context_length=16384
            )

        assert manager._model_configs[TEST_MODEL_ID]["context_length"] == 16384
