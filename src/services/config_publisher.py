"""Redis Pub/Sub publisher for model configuration changes.

LLM Operations Mesh - Phase 2: Real-time config sync across services.

This module publishes MODEL_CONFIG_CHANGED events when models are loaded/unloaded,
allowing CMS and other services to update their caches immediately.

Reference: LLM_OPERATIONS_MESH_ARCHITECTURE.md - Phase 2
"""

import json
import logging
from datetime import datetime, timezone
from typing import Any

import redis.asyncio as redis
from redis.asyncio.client import Redis
from redis.exceptions import ConnectionError, RedisError

logger = logging.getLogger(__name__)

# Redis channels
CHANNEL_MODEL_CONFIG = "model:config:changes"
CHANNEL_MODEL_LIFECYCLE = "model:lifecycle:events"

# Cache key patterns
CACHE_KEY_MODEL_CONFIG = "model:config:{model_id}"
CACHE_KEY_MODELS_AVAILABLE = "models:available"
CACHE_KEY_MODEL_USAGE = "model:usage:{model_id}"

# Default TTL for cache entries (1 hour)
DEFAULT_CACHE_TTL = 3600

# TTL for usage tracking keys (24 hours)
USAGE_TTL = 86400


class ConfigPublisher:
    """Publishes model configuration changes to Redis.
    
    Responsibilities:
    - Publish MODEL_CONFIG_CHANGED events on load/unload
    - Write model config to Redis cache for other services
    - Maintain models:available set
    
    Example:
        publisher = ConfigPublisher("redis://localhost:6379")
        await publisher.connect()
        await publisher.publish_model_loaded("qwen3-8b", context_length=2048)
        await publisher.close()
    """
    
    def __init__(self, redis_url: str = "redis://localhost:6379"):
        """Initialize publisher.
        
        Args:
            redis_url: Redis connection URL
        """
        self.redis_url = redis_url
        self._redis: Redis | None = None
        self._connected = False
    
    @property
    def is_connected(self) -> bool:
        """Check if connected to Redis."""
        return self._connected and self._redis is not None
    
    async def connect(self) -> bool:
        """Establish Redis connection.
        
        Returns:
            True if connected, False otherwise
        """
        try:
            self._redis = redis.from_url(
                self.redis_url,
                encoding="utf-8",
                decode_responses=True,
            )
            await self._redis.ping()
            self._connected = True
            logger.info(f"ConfigPublisher connected to Redis at {self.redis_url}")
            return True
        except (ConnectionError, RedisError) as e:
            logger.warning(f"ConfigPublisher failed to connect to Redis: {e}")
            self._connected = False
            return False
    
    async def close(self) -> None:
        """Close Redis connection."""
        if self._redis:
            await self._redis.close()
            self._connected = False
            logger.info("ConfigPublisher disconnected from Redis")
    
    async def _get_usage_stats(self, model_id: str) -> dict[str, Any] | None:
        """Read usage stats from model:usage:{model_id} for lifecycle events.

        Returns:
            Dict with last_used and request_count, or None if unavailable.
        """
        if not self.is_connected:
            return None
        try:
            usage_key = CACHE_KEY_MODEL_USAGE.format(model_id=model_id)
            data = await self._redis.hgetall(usage_key)
            if not data:
                return None
            return {
                "last_used": data.get("last_used"),
                "request_count": int(data.get("request_count", 0)),
            }
        except (ConnectionError, RedisError):
            return None

    async def publish_model_loaded(
        self,
        model_id: str,
        context_length: int,
        memory_mb: int = 0,
        roles: list[str] | None = None,
        trigger: str = "api_request",
        **extra: Any,
    ) -> bool:
        """Publish MODEL_LOADED event and update cache.

        Publishes to both model:config:changes (existing) and
        model:lifecycle:events (Phase B) channels.
        
        Args:
            model_id: Model identifier
            context_length: Context window size
            memory_mb: Memory usage in MB
            roles: Model roles (e.g., ["chat", "code"])
            trigger: What triggered the load ("api_request", "preset_load", "warmup")
            **extra: Additional model metadata
            
        Returns:
            True if published successfully
        """
        if not self.is_connected:
            logger.debug("ConfigPublisher not connected, skipping publish")
            return False
        
        try:
            now = datetime.now(timezone.utc).isoformat()

            # Build config event payload (existing behaviour)
            event = {
                "event_type": "MODEL_LOADED",
                "model_id": model_id,
                "data": {
                    "context_length": context_length,
                    "status": "loaded",
                    "memory_mb": memory_mb,
                    "roles": roles or [],
                    **extra,
                },
                "source": "inference-service",
                "timestamp": now,
            }
            
            # Publish to config channel (existing)
            await self._redis.publish(CHANNEL_MODEL_CONFIG, json.dumps(event))
            
            # Phase B: Publish richer lifecycle event (AC-B.1, AC-B.3)
            usage_stats = await self._get_usage_stats(model_id)
            lifecycle_event = {
                "event_type": "MODEL_LOADED",
                "model_id": model_id,
                "timestamp": now,
                "source": "inference-service",
                "memory_mb": memory_mb,
                "trigger": trigger,
                "usage_stats": usage_stats,
            }
            await self._redis.publish(
                CHANNEL_MODEL_LIFECYCLE, json.dumps(lifecycle_event)
            )
            
            # Update cache
            cache_key = CACHE_KEY_MODEL_CONFIG.format(model_id=model_id)
            await self._redis.setex(
                cache_key,
                DEFAULT_CACHE_TTL,
                json.dumps(event["data"]),
            )
            
            # Add to available models set
            await self._redis.sadd(CACHE_KEY_MODELS_AVAILABLE, model_id)
            
            logger.info(
                f"Published MODEL_LOADED for {model_id}",
                extra={"context_length": context_length, "trigger": trigger},
            )
            return True
            
        except (ConnectionError, RedisError) as e:
            logger.warning(f"Failed to publish MODEL_LOADED: {e}")
            self._connected = False
            return False
    
    async def publish_model_unloaded(
        self, model_id: str, trigger: str = "api_request"
    ) -> bool:
        """Publish MODEL_UNLOADED event and update cache.

        Publishes to both model:config:changes (existing) and
        model:lifecycle:events (Phase B) channels.
        
        Args:
            model_id: Model identifier
            trigger: What triggered the unload ("api_request", "eviction", "shutdown")
            
        Returns:
            True if published successfully
        """
        if not self.is_connected:
            logger.debug("ConfigPublisher not connected, skipping publish")
            return False
        
        try:
            now = datetime.now(timezone.utc).isoformat()

            # Read usage stats before unload (lifecycle event enrichment)
            usage_stats = await self._get_usage_stats(model_id)

            # Build config event payload (existing behaviour)
            event = {
                "event_type": "MODEL_UNLOADED",
                "model_id": model_id,
                "data": {
                    "status": "unloaded",
                },
                "source": "inference-service",
                "timestamp": now,
            }
            
            # Publish to config channel (existing)
            await self._redis.publish(CHANNEL_MODEL_CONFIG, json.dumps(event))

            # Phase B: Publish richer lifecycle event (AC-B.2, AC-B.3)
            lifecycle_event = {
                "event_type": "MODEL_UNLOADED",
                "model_id": model_id,
                "timestamp": now,
                "source": "inference-service",
                "memory_mb": 0,
                "trigger": trigger,
                "usage_stats": usage_stats,
            }
            await self._redis.publish(
                CHANNEL_MODEL_LIFECYCLE, json.dumps(lifecycle_event)
            )
            
            # Remove from cache
            cache_key = CACHE_KEY_MODEL_CONFIG.format(model_id=model_id)
            await self._redis.delete(cache_key)
            
            # Remove from available models set
            await self._redis.srem(CACHE_KEY_MODELS_AVAILABLE, model_id)
            
            logger.info(
                f"Published MODEL_UNLOADED for {model_id}",
                extra={"trigger": trigger},
            )
            return True
            
        except (ConnectionError, RedisError) as e:
            logger.warning(f"Failed to publish MODEL_UNLOADED: {e}")
            self._connected = False
            return False

    async def record_model_usage(self, model_id: str) -> bool:
        """Record model usage for LRU eviction decisions.

        Writes a Redis HASH at model:usage:{model_id} with:
        - last_used: ISO 8601 timestamp of this request
        - request_count: incremented on each call

        The key TTL is refreshed to 24 hours on every write to prevent
        stale data accumulation for models that are no longer in use.

        LLM Operations Mesh — Phase A (WBS-MESH-A, AC-A.1, AC-A.3, AC-A.4).

        Args:
            model_id: Model identifier that served the request.

        Returns:
            True if recorded successfully, False on Redis failure.
        """
        if not self.is_connected:
            logger.debug("ConfigPublisher not connected, skipping usage tracking")
            return False

        try:
            usage_key = CACHE_KEY_MODEL_USAGE.format(model_id=model_id)
            now = datetime.now(timezone.utc).isoformat()

            # Atomically set last_used and increment request_count
            pipe = self._redis.pipeline()
            pipe.hset(usage_key, "last_used", now)
            pipe.hincrby(usage_key, "request_count", 1)
            pipe.expire(usage_key, USAGE_TTL)
            await pipe.execute()

            logger.debug(
                f"Recorded usage for {model_id}",
                extra={"model_id": model_id, "last_used": now},
            )
            return True

        except (ConnectionError, RedisError) as e:
            # AC-A.4: Graceful fallback — log warning, don't fail inference
            logger.warning(f"Failed to record model usage for {model_id}: {e}")
            return False


# Singleton instance
_publisher: ConfigPublisher | None = None


def get_config_publisher() -> ConfigPublisher | None:
    """Get singleton ConfigPublisher instance.
    
    Returns:
        ConfigPublisher if initialized, None otherwise
    """
    return _publisher


async def initialize_publisher(redis_url: str) -> ConfigPublisher:
    """Initialize and connect the singleton ConfigPublisher.
    
    Args:
        redis_url: Redis connection URL
        
    Returns:
        Connected ConfigPublisher instance
    """
    global _publisher
    _publisher = ConfigPublisher(redis_url)
    await _publisher.connect()
    return _publisher


async def shutdown_publisher() -> None:
    """Shutdown the singleton ConfigPublisher."""
    global _publisher
    if _publisher:
        await _publisher.close()
        _publisher = None
