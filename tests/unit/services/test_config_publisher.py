"""Unit tests for ConfigPublisher — including Phase A usage tracking.

Tests:
- record_model_usage() writes HASH with correct fields + TTL (AC-A.1, AC-A.3)
- Graceful fallback when Redis is unavailable (AC-A.4)
- publish_model_loaded / publish_model_unloaded (existing behaviour)

Reference: WBS-MESH-A Tasks A.1, A.7, A.9
"""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.services.config_publisher import (
    CACHE_KEY_MODEL_USAGE,
    CHANNEL_MODEL_CONFIG,
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
        publisher._redis.publish.assert_awaited_once()
        call_args = publisher._redis.publish.call_args
        assert call_args[0][0] == CHANNEL_MODEL_CONFIG

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
        publisher._redis.publish.assert_awaited_once()

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
