"""Unit tests for ConfigPublisher — Phase A usage tracking + Phase B lifecycle events.

Tests:
- record_model_usage() writes HASH with correct fields + TTL (AC-A.1, AC-A.3)
- Graceful fallback when Redis is unavailable (AC-A.4)
- publish_model_loaded / publish_model_unloaded publish to BOTH channels (AC-B.1, AC-B.2)
- Lifecycle events include richer metadata (AC-B.3)
- CHANNEL_MODEL_LIFECYCLE constant is used (AC-B.4)

Reference: WBS-MESH-A Tasks A.1, A.7, A.9; WBS-MESH-B Tasks B.1-B.6
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.services.config_publisher import (
    CACHE_KEY_MODEL_USAGE,
    CHANNEL_MODEL_CONFIG,
    CHANNEL_MODEL_LIFECYCLE,
    ConfigPublisher,
    USAGE_TTL,
    get_config_publisher,
    initialize_publisher,
    shutdown_publisher,
)


# =============================================================================
# Constants
# =============================================================================

TEST_MODEL_ID = "qwen3-8b"
TEST_REDIS_URL = "redis://localhost:6379"


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def publisher() -> ConfigPublisher:
    """Create a ConfigPublisher with a mocked Redis client."""
    pub = ConfigPublisher(TEST_REDIS_URL)
    # Mock the Redis client so tests don't require a running Redis
    pub._redis = AsyncMock()
    pub._connected = True
    # Pipeline mock: pipeline() returns an object with hset, hincrby, expire, execute
    mock_pipeline = AsyncMock()
    mock_pipeline.hset = MagicMock(return_value=mock_pipeline)
    mock_pipeline.hincrby = MagicMock(return_value=mock_pipeline)
    mock_pipeline.expire = MagicMock(return_value=mock_pipeline)
    mock_pipeline.execute = AsyncMock(return_value=[True, 1, True])
    pub._redis.pipeline = MagicMock(return_value=mock_pipeline)
    # Default: hgetall returns empty dict (no usage data yet)
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
# TestRecordModelUsage — AC-A.1: HASH fields, AC-A.3: TTL, AC-A.4: fallback
# =============================================================================


class TestRecordModelUsage:
    """Tests for record_model_usage() — Phase A usage tracking."""

    @pytest.mark.asyncio
    async def test_writes_hash_with_correct_fields(
        self, publisher: ConfigPublisher
    ) -> None:
        """AC-A.1: record_model_usage writes last_used and increments request_count."""
        result = await publisher.record_model_usage(TEST_MODEL_ID)

        assert result is True

        pipe = publisher._redis.pipeline()
        expected_key = CACHE_KEY_MODEL_USAGE.format(model_id=TEST_MODEL_ID)

        # Verify hset was called with last_used (ISO 8601 timestamp)
        pipe.hset.assert_called_once()
        call_args = pipe.hset.call_args
        assert call_args[0][0] == expected_key
        assert call_args[0][1] == "last_used"
        # Verify the timestamp is a valid ISO 8601 string
        timestamp_str = call_args[0][2]
        parsed = datetime.fromisoformat(timestamp_str)
        assert parsed.tzinfo is not None  # Must be timezone-aware

        # Verify hincrby was called to increment request_count
        pipe.hincrby.assert_called_once_with(expected_key, "request_count", 1)

    @pytest.mark.asyncio
    async def test_sets_ttl_to_86400(self, publisher: ConfigPublisher) -> None:
        """AC-A.3: Usage keys have 24-hour TTL refreshed on each write."""
        await publisher.record_model_usage(TEST_MODEL_ID)

        pipe = publisher._redis.pipeline()
        expected_key = CACHE_KEY_MODEL_USAGE.format(model_id=TEST_MODEL_ID)
        pipe.expire.assert_called_once_with(expected_key, USAGE_TTL)

    @pytest.mark.asyncio
    async def test_pipeline_executes_atomically(
        self, publisher: ConfigPublisher
    ) -> None:
        """Pipeline execute() is called to run all commands atomically."""
        await publisher.record_model_usage(TEST_MODEL_ID)

        pipe = publisher._redis.pipeline()
        pipe.execute.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_returns_false_when_disconnected(
        self, disconnected_publisher: ConfigPublisher
    ) -> None:
        """AC-A.4: Returns False when not connected to Redis."""
        result = await disconnected_publisher.record_model_usage(TEST_MODEL_ID)
        assert result is False

    @pytest.mark.asyncio
    async def test_graceful_fallback_on_redis_error(
        self, publisher: ConfigPublisher
    ) -> None:
        """AC-A.4: Redis errors are caught — returns False, no exception."""
        from redis.exceptions import ConnectionError as RedisConnectionError

        pipe = publisher._redis.pipeline()
        pipe.execute = AsyncMock(side_effect=RedisConnectionError("connection lost"))

        result = await publisher.record_model_usage(TEST_MODEL_ID)
        assert result is False

    @pytest.mark.asyncio
    async def test_timestamp_is_utc(self, publisher: ConfigPublisher) -> None:
        """Timestamp uses UTC timezone."""
        await publisher.record_model_usage(TEST_MODEL_ID)

        pipe = publisher._redis.pipeline()
        call_args = pipe.hset.call_args
        timestamp_str = call_args[0][2]
        parsed = datetime.fromisoformat(timestamp_str)
        assert parsed.tzinfo == timezone.utc

    @pytest.mark.asyncio
    async def test_multiple_calls_increment_count(
        self, publisher: ConfigPublisher
    ) -> None:
        """Multiple calls each invoke hincrby(request_count, 1)."""
        await publisher.record_model_usage(TEST_MODEL_ID)
        await publisher.record_model_usage(TEST_MODEL_ID)

        pipe = publisher._redis.pipeline()
        assert pipe.hincrby.call_count == 2

    @pytest.mark.asyncio
    async def test_different_models_use_different_keys(
        self, publisher: ConfigPublisher
    ) -> None:
        """Each model_id gets its own usage key."""
        await publisher.record_model_usage("model-a")
        await publisher.record_model_usage("model-b")

        pipe = publisher._redis.pipeline()
        calls = pipe.hset.call_args_list
        keys = {c[0][0] for c in calls}
        assert "model:usage:model-a" in keys
        assert "model:usage:model-b" in keys


# =============================================================================
# TestPublishModelLoaded
# =============================================================================


class TestPublishModelLoaded:
    """Tests for publish_model_loaded()."""

    @pytest.mark.asyncio
    async def test_publishes_to_config_channel(
        self, publisher: ConfigPublisher
    ) -> None:
        """Publish event to model:config:changes channel."""
        publisher._redis.publish = AsyncMock(return_value=1)
        publisher._redis.setex = AsyncMock()
        publisher._redis.sadd = AsyncMock()

        result = await publisher.publish_model_loaded(
            model_id=TEST_MODEL_ID,
            context_length=8192,
            memory_mb=4096,
        )

        assert result is True
        # Phase B: publish is called TWICE — config channel + lifecycle channel
        assert publisher._redis.publish.await_count == 2
        first_call = publisher._redis.publish.call_args_list[0]
        assert first_call[0][0] == CHANNEL_MODEL_CONFIG

    @pytest.mark.asyncio
    async def test_returns_false_when_disconnected(
        self, disconnected_publisher: ConfigPublisher
    ) -> None:
        """Returns False when not connected."""
        result = await disconnected_publisher.publish_model_loaded(
            model_id=TEST_MODEL_ID, context_length=8192
        )
        assert result is False


# =============================================================================
# TestPublishModelUnloaded
# =============================================================================


class TestPublishModelUnloaded:
    """Tests for publish_model_unloaded()."""

    @pytest.mark.asyncio
    async def test_publishes_to_config_channel(
        self, publisher: ConfigPublisher
    ) -> None:
        """Publish unload event to model:config:changes channel."""
        publisher._redis.publish = AsyncMock(return_value=1)
        publisher._redis.delete = AsyncMock()
        publisher._redis.srem = AsyncMock()

        result = await publisher.publish_model_unloaded(TEST_MODEL_ID)

        assert result is True
        # Phase B: publish is called TWICE — config channel + lifecycle channel
        assert publisher._redis.publish.await_count == 2
        first_call = publisher._redis.publish.call_args_list[0]
        assert first_call[0][0] == CHANNEL_MODEL_CONFIG

    @pytest.mark.asyncio
    async def test_returns_false_when_disconnected(
        self, disconnected_publisher: ConfigPublisher
    ) -> None:
        """Returns False when not connected."""
        result = await disconnected_publisher.publish_model_unloaded(
            TEST_MODEL_ID
        )
        assert result is False


# =============================================================================
# TestLifecycleEvents — Phase B (AC-B.1, AC-B.2, AC-B.3, AC-B.4)
# =============================================================================


class TestLifecycleEventsLoaded:
    """Tests for lifecycle event publishing on model load — Phase B."""

    @pytest.mark.asyncio
    async def test_publishes_to_lifecycle_channel(
        self, publisher: ConfigPublisher
    ) -> None:
        """AC-B.1: publish_model_loaded publishes to CHANNEL_MODEL_LIFECYCLE."""
        publisher._redis.publish = AsyncMock(return_value=1)
        publisher._redis.setex = AsyncMock()
        publisher._redis.sadd = AsyncMock()

        await publisher.publish_model_loaded(
            model_id=TEST_MODEL_ID,
            context_length=8192,
            memory_mb=4096,
        )

        # Second publish call goes to lifecycle channel
        second_call = publisher._redis.publish.call_args_list[1]
        assert second_call[0][0] == CHANNEL_MODEL_LIFECYCLE

    @pytest.mark.asyncio
    async def test_lifecycle_event_has_required_fields(
        self, publisher: ConfigPublisher
    ) -> None:
        """AC-B.3: Lifecycle event includes event_type, model_id, timestamp, source, memory_mb, trigger."""
        publisher._redis.publish = AsyncMock(return_value=1)
        publisher._redis.setex = AsyncMock()
        publisher._redis.sadd = AsyncMock()

        await publisher.publish_model_loaded(
            model_id=TEST_MODEL_ID,
            context_length=8192,
            memory_mb=4096,
            trigger="warmup",
        )

        second_call = publisher._redis.publish.call_args_list[1]
        payload = json.loads(second_call[0][1])

        assert payload["event_type"] == "MODEL_LOADED"
        assert payload["model_id"] == TEST_MODEL_ID
        assert payload["source"] == "inference-service"
        assert payload["memory_mb"] == 4096
        assert payload["trigger"] == "warmup"
        # Timestamp must be valid ISO 8601
        parsed = datetime.fromisoformat(payload["timestamp"])
        assert parsed.tzinfo is not None

    @pytest.mark.asyncio
    async def test_lifecycle_event_includes_usage_stats(
        self, publisher: ConfigPublisher
    ) -> None:
        """AC-B.3: Lifecycle event includes usage_stats from Phase A data."""
        publisher._redis.publish = AsyncMock(return_value=1)
        publisher._redis.setex = AsyncMock()
        publisher._redis.sadd = AsyncMock()
        publisher._redis.hgetall = AsyncMock(return_value={
            "last_used": "2026-02-12T14:30:00+00:00",
            "request_count": "42",
        })

        await publisher.publish_model_loaded(
            model_id=TEST_MODEL_ID,
            context_length=8192,
            memory_mb=4096,
        )

        second_call = publisher._redis.publish.call_args_list[1]
        payload = json.loads(second_call[0][1])

        assert payload["usage_stats"] is not None
        assert payload["usage_stats"]["last_used"] == "2026-02-12T14:30:00+00:00"
        assert payload["usage_stats"]["request_count"] == 42

    @pytest.mark.asyncio
    async def test_lifecycle_event_usage_stats_null_when_no_data(
        self, publisher: ConfigPublisher
    ) -> None:
        """usage_stats is None when model has no usage data."""
        publisher._redis.publish = AsyncMock(return_value=1)
        publisher._redis.setex = AsyncMock()
        publisher._redis.sadd = AsyncMock()
        publisher._redis.hgetall = AsyncMock(return_value={})

        await publisher.publish_model_loaded(
            model_id=TEST_MODEL_ID,
            context_length=8192,
            memory_mb=4096,
        )

        second_call = publisher._redis.publish.call_args_list[1]
        payload = json.loads(second_call[0][1])
        assert payload["usage_stats"] is None

    @pytest.mark.asyncio
    async def test_default_trigger_is_api_request(
        self, publisher: ConfigPublisher
    ) -> None:
        """Default trigger for publish_model_loaded is 'api_request'."""
        publisher._redis.publish = AsyncMock(return_value=1)
        publisher._redis.setex = AsyncMock()
        publisher._redis.sadd = AsyncMock()

        await publisher.publish_model_loaded(
            model_id=TEST_MODEL_ID,
            context_length=8192,
        )

        second_call = publisher._redis.publish.call_args_list[1]
        payload = json.loads(second_call[0][1])
        assert payload["trigger"] == "api_request"


class TestLifecycleEventsUnloaded:
    """Tests for lifecycle event publishing on model unload — Phase B."""

    @pytest.mark.asyncio
    async def test_publishes_to_lifecycle_channel(
        self, publisher: ConfigPublisher
    ) -> None:
        """AC-B.2: publish_model_unloaded publishes to CHANNEL_MODEL_LIFECYCLE."""
        publisher._redis.publish = AsyncMock(return_value=1)
        publisher._redis.delete = AsyncMock()
        publisher._redis.srem = AsyncMock()

        await publisher.publish_model_unloaded(TEST_MODEL_ID)

        second_call = publisher._redis.publish.call_args_list[1]
        assert second_call[0][0] == CHANNEL_MODEL_LIFECYCLE

    @pytest.mark.asyncio
    async def test_lifecycle_unload_has_required_fields(
        self, publisher: ConfigPublisher
    ) -> None:
        """AC-B.3: Unload lifecycle event includes all required metadata."""
        publisher._redis.publish = AsyncMock(return_value=1)
        publisher._redis.delete = AsyncMock()
        publisher._redis.srem = AsyncMock()

        await publisher.publish_model_unloaded(
            TEST_MODEL_ID, trigger="eviction"
        )

        second_call = publisher._redis.publish.call_args_list[1]
        payload = json.loads(second_call[0][1])

        assert payload["event_type"] == "MODEL_UNLOADED"
        assert payload["model_id"] == TEST_MODEL_ID
        assert payload["source"] == "inference-service"
        assert payload["trigger"] == "eviction"
        assert "usage_stats" in payload
        parsed = datetime.fromisoformat(payload["timestamp"])
        assert parsed.tzinfo is not None

    @pytest.mark.asyncio
    async def test_unload_lifecycle_includes_usage_stats(
        self, publisher: ConfigPublisher
    ) -> None:
        """AC-B.3: Unload lifecycle event includes usage_stats from Phase A."""
        publisher._redis.publish = AsyncMock(return_value=1)
        publisher._redis.delete = AsyncMock()
        publisher._redis.srem = AsyncMock()
        publisher._redis.hgetall = AsyncMock(return_value={
            "last_used": "2026-02-12T15:00:00+00:00",
            "request_count": "7",
        })

        await publisher.publish_model_unloaded(TEST_MODEL_ID)

        second_call = publisher._redis.publish.call_args_list[1]
        payload = json.loads(second_call[0][1])

        assert payload["usage_stats"]["last_used"] == "2026-02-12T15:00:00+00:00"
        assert payload["usage_stats"]["request_count"] == 7

    @pytest.mark.asyncio
    async def test_default_unload_trigger_is_api_request(
        self, publisher: ConfigPublisher
    ) -> None:
        """Default trigger for publish_model_unloaded is 'api_request'."""
        publisher._redis.publish = AsyncMock(return_value=1)
        publisher._redis.delete = AsyncMock()
        publisher._redis.srem = AsyncMock()

        await publisher.publish_model_unloaded(TEST_MODEL_ID)

        second_call = publisher._redis.publish.call_args_list[1]
        payload = json.loads(second_call[0][1])
        assert payload["trigger"] == "api_request"


class TestGetUsageStats:
    """Tests for _get_usage_stats() helper."""

    @pytest.mark.asyncio
    async def test_returns_usage_data(self, publisher: ConfigPublisher) -> None:
        """Returns dict with last_used and request_count from Redis."""
        publisher._redis.hgetall = AsyncMock(return_value={
            "last_used": "2026-02-12T14:30:00+00:00",
            "request_count": "42",
        })

        result = await publisher._get_usage_stats(TEST_MODEL_ID)

        assert result == {
            "last_used": "2026-02-12T14:30:00+00:00",
            "request_count": 42,
        }

    @pytest.mark.asyncio
    async def test_returns_none_when_no_data(
        self, publisher: ConfigPublisher
    ) -> None:
        """Returns None when key doesn't exist (empty hgetall)."""
        publisher._redis.hgetall = AsyncMock(return_value={})

        result = await publisher._get_usage_stats(TEST_MODEL_ID)
        assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_when_disconnected(
        self, disconnected_publisher: ConfigPublisher
    ) -> None:
        """Returns None when not connected to Redis."""
        result = await disconnected_publisher._get_usage_stats(TEST_MODEL_ID)
        assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_on_redis_error(
        self, publisher: ConfigPublisher
    ) -> None:
        """Graceful fallback on Redis error."""
        from redis.exceptions import ConnectionError as RedisConnectionError

        publisher._redis.hgetall = AsyncMock(
            side_effect=RedisConnectionError("connection lost")
        )

        result = await publisher._get_usage_stats(TEST_MODEL_ID)
        assert result is None


# =============================================================================
# TestSingleton
# =============================================================================


class TestSingleton:
    """Tests for singleton lifecycle functions."""

    @pytest.mark.asyncio
    async def test_initialize_and_get(self) -> None:
        """initialize_publisher creates a singleton retrievable by get_config_publisher."""
        with patch(
            "src.services.config_publisher.redis.from_url"
        ) as mock_from_url:
            mock_client = AsyncMock()
            mock_client.ping = AsyncMock()
            mock_from_url.return_value = mock_client

            pub = await initialize_publisher(TEST_REDIS_URL)
            assert pub is not None
            assert get_config_publisher() is pub

            await shutdown_publisher()
            assert get_config_publisher() is None
