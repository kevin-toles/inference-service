"""Caching infrastructure for inference-service.

Implements WBS-INF16: Caching Infrastructure

AC-16.1: InferenceCache ABC with required methods
AC-16.2: PromptCache caches tokenized prompts
AC-16.3: HandoffCache with asyncio.Lock (AP-10.1)
AC-16.4: CompressionCache for compressed content
AC-16.5: CacheInvalidator invalidates on model change

Reference: ARCHITECTURE.md → Caching Strategy

Per CODING_PATTERNS_ANALYSIS AP-10.1:
- Async caches use per-resource locks to prevent race conditions
"""

import asyncio
import hashlib
import json
from abc import ABC, abstractmethod
from dataclasses import asdict
from typing import Any, Protocol, runtime_checkable

from src.orchestration.context import HandoffState


# =============================================================================
# Type Definitions
# =============================================================================


@runtime_checkable
class Tokenizer(Protocol):
    """Protocol for tokenizer interface."""

    def tokenize(self, text: bytes) -> list[int]:
        """Tokenize text into token IDs."""
        ...


@runtime_checkable
class AsyncRedisClient(Protocol):
    """Protocol for async Redis client."""

    async def get(self, key: str) -> bytes | None:
        """Get value from Redis."""
        ...

    async def setex(self, key: str, ttl: int, value: str) -> None:
        """Set value with expiration."""
        ...


# =============================================================================
# InferenceCache ABC (AC-16.1)
# =============================================================================


class InferenceCache(ABC):
    """Abstract base class for inference caching.

    All cache implementations must implement this interface to ensure
    consistent behavior and support cache invalidation.

    Note: The get/store methods use **kwargs to allow subclasses to
    define domain-specific parameters while maintaining a consistent
    interface for cache invalidation and clearing.

    Methods:
        get: Retrieve cached value
        store: Store value in cache
        clear: Clear entire cache
        invalidate_by_model: Clear entries for specific model
    """

    @abstractmethod
    def get(self, **kwargs: Any) -> Any:
        """Retrieve cached value.

        Subclasses define specific parameters (e.g., key, request_id, content).

        Returns:
            Cached value or None if not found.
        """
        ...

    @abstractmethod
    def store(self, **kwargs: Any) -> None:
        """Store value in cache.

        Subclasses define specific parameters (e.g., key, value, state, ttl).
        """
        ...

    @abstractmethod
    def clear(self) -> None:
        """Clear entire cache."""
        ...

    @abstractmethod
    def invalidate_by_model(self, model_id: str) -> None:
        """Invalidate cache entries for specific model.

        Args:
            model_id: Model ID to invalidate entries for.
        """
        ...


# =============================================================================
# PromptCache (AC-16.2)
# =============================================================================


class PromptCache(InferenceCache):
    """Cache tokenized prompts since they rarely change.

    Implements AC-16.2: Caches tokenized prompts for faster subsequent access.

    Attributes:
        tokenizer: Tokenizer instance for converting text to tokens.
        model_id: Optional model ID for invalidation filtering.

    Example:
        cache = PromptCache(tokenizer=llama_model, model_id="phi-4")
        tokens = cache.get_tokenized("System prompt here")
        # Second call uses cache, no re-tokenization
        tokens = cache.get_tokenized("System prompt here")
    """

    def __init__(
        self,
        tokenizer: Tokenizer,
        model_id: str | None = None,
    ) -> None:
        """Initialize PromptCache.

        Args:
            tokenizer: Tokenizer for converting text to tokens.
            model_id: Optional model ID for selective invalidation.
        """
        self._tokenizer = tokenizer
        self._model_id = model_id
        self._cache: dict[str, list[int]] = {}

    def get_tokenized(self, text: str) -> list[int]:
        """Get tokenized version of text, using cache if available.

        Args:
            text: Text to tokenize.

        Returns:
            List of token IDs.
        """
        key = hashlib.md5(text.encode()).hexdigest()
        if key not in self._cache:
            self._cache[key] = self._tokenizer.tokenize(text.encode())
        return self._cache[key]

    def get(self, **kwargs: Any) -> list[int] | None:
        """Retrieve cached tokens by key.

        Args:
            key: Cache key.

        Returns:
            Cached token list or None.
        """
        key = kwargs.get("key", "")
        return self._cache.get(key)

    def store(self, **kwargs: Any) -> None:
        """Store tokenized value in cache.

        Args:
            key: Cache key.
            value: Token list to cache.
        """
        key = kwargs.get("key", "")
        value = kwargs.get("value")
        if key and value is not None:
            self._cache[key] = value

    def clear(self) -> None:
        """Clear entire token cache."""
        self._cache.clear()

    def invalidate_by_model(self, model_id: str) -> None:
        """Invalidate cache if model matches.

        Args:
            model_id: Model ID to check against.
        """
        if self._model_id is None or self._model_id == model_id:
            self.clear()


# =============================================================================
# HandoffCache (AC-16.3)
# =============================================================================


class HandoffCache(InferenceCache):
    """Cache structured handoff state between pipeline steps.

    Implements AC-16.3: HandoffCache with asyncio.Lock (AP-10.1).

    Uses per-resource locks per CODING_PATTERNS_ANALYSIS AP-10.1
    to prevent race conditions in async context.

    Attributes:
        _redis: Optional Redis client for distributed caching.
        _local: Local dict fallback for development.
        _locks: Per-resource asyncio.Lock instances.

    Example:
        cache = HandoffCache()
        await cache.store(HandoffState(request_id="req-123"))
        state = await cache.get("req-123")
    """

    def __init__(
        self,
        redis_client: AsyncRedisClient | None = None,
    ) -> None:
        """Initialize HandoffCache.

        Args:
            redis_client: Optional async Redis client. Falls back to
                local dict if not provided.
        """
        self._redis = redis_client
        self._local: dict[str, str] = {}
        self._locks: dict[str, asyncio.Lock] = {}

    def _get_lock(self, key: str) -> asyncio.Lock:
        """Get or create lock for specific key.

        Per AP-10.1: Use per-resource locks for async caches.

        Args:
            key: Cache key to get lock for.

        Returns:
            asyncio.Lock for the specified key.
        """
        if key not in self._locks:
            self._locks[key] = asyncio.Lock()
        return self._locks[key]

    async def store(
        self,
        state: HandoffState | None = None,
        ttl: int = 3600,
        **kwargs: Any,
    ) -> None:
        """Store handoff state with optional TTL.

        Can be called async with HandoffState or sync via interface.

        Args:
            state: HandoffState to store (async usage).
            ttl: Time-to-live in seconds (default 3600).
            **kwargs: key/value for interface compliance.
        """
        key = kwargs.get("key")
        value = kwargs.get("value")
        
        if state is not None:
            cache_key = f"handoff:{state.request_id}"
            async with self._get_lock(cache_key):
                data = json.dumps(asdict(state))
                if self._redis:
                    await self._redis.setex(cache_key, ttl, data)
                else:
                    self._local[cache_key] = data
        elif key is not None and value is not None:
            # Sync interface fallback
            self._local[key] = value

    async def get(
        self,
        request_id: str | None = None,
        **kwargs: Any,
    ) -> HandoffState | str | None:
        """Retrieve handoff state by request ID.

        Args:
            request_id: Request ID to lookup (async usage).
            **kwargs: key for interface compliance.

        Returns:
            HandoffState if found via request_id, str if via key, None otherwise.
        """
        key = kwargs.get("key")
        
        if request_id is not None:
            cache_key = f"handoff:{request_id}"
            async with self._get_lock(cache_key):
                if self._redis:
                    redis_data = await self._redis.get(cache_key)
                    if redis_data:
                        return HandoffState(**json.loads(redis_data))
                    return None
                else:
                    local_data = self._local.get(cache_key)
                    return HandoffState(**json.loads(local_data)) if local_data else None
        elif key is not None:
            # Sync interface fallback
            return self._local.get(key)
        return None

    def clear(self) -> None:
        """Clear all handoff states."""
        self._local.clear()
        self._locks.clear()

    def invalidate_by_model(self, _model_id: str) -> None:
        """Invalidate all handoff states.

        Handoff states are model-agnostic but cleared for safety.

        Args:
            _model_id: Model ID (ignored, clears all).
        """
        self.clear()


# =============================================================================
# CompressionCache (AC-16.4)
# =============================================================================


class CompressionCache(InferenceCache):
    """Cache compressed versions of content.

    Implements AC-16.4: CompressionCache for compressed content.

    Stores compressed content keyed by original content hash and
    target token count for efficient retrieval.

    Example:
        cache = CompressionCache()
        cache.store("long content...", 100, "compressed version")
        compressed = cache.get("long content...", 100)
    """

    def __init__(self) -> None:
        """Initialize CompressionCache."""
        self._cache: dict[str, str] = {}

    def get_key(self, content: str, target_tokens: int) -> str:
        """Generate cache key from content and target tokens.

        Args:
            content: Original content.
            target_tokens: Target token count for compression.

        Returns:
            Cache key string.
        """
        content_hash = hashlib.md5(content.encode()).hexdigest()
        return f"{content_hash}:{target_tokens}"

    def get(
        self,
        content: str | None = None,
        target_tokens: int | None = None,
        **kwargs: Any,
    ) -> str | None:
        """Retrieve cached compression.

        Can lookup by content+target_tokens or by direct key.

        Args:
            content: Original content (optional).
            target_tokens: Target token count (optional).
            **kwargs: key for direct lookup.

        Returns:
            Cached compressed content or None.
        """
        key = kwargs.get("key")
        if key:
            return self._cache.get(key)
        if content is not None and target_tokens is not None:
            lookup_key = self.get_key(content, target_tokens)
            return self._cache.get(lookup_key)
        return None

    def store(
        self,
        content: str | None = None,
        target_tokens: int | None = None,
        compressed: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Store compressed content.

        Can store by content+target_tokens+compressed or by key+value.

        Args:
            content: Original content (optional).
            target_tokens: Target token count (optional).
            compressed: Compressed version (optional).
            **kwargs: key/value for direct storage.
        """
        key = kwargs.get("key")
        value = kwargs.get("value")
        
        if key and value:
            self._cache[key] = value
        elif content is not None and target_tokens is not None and compressed:
            store_key = self.get_key(content, target_tokens)
            self._cache[store_key] = compressed

    def clear(self) -> None:
        """Clear entire compression cache."""
        self._cache.clear()

    def invalidate_by_model(self, _model_id: str) -> None:
        """Invalidate all compressed content.

        Compression depends on model tokenization, so clear all.

        Args:
            _model_id: Model ID (ignored, clears all).
        """
        self.clear()


# =============================================================================
# CacheInvalidator (AC-16.5)
# =============================================================================


class CacheInvalidator:
    """Invalidate caches when model configuration changes.

    Implements AC-16.5: Invalidates on model change.

    Tracks model versions and triggers cache invalidation when
    models are reloaded with different hashes.

    Attributes:
        caches: List of caches to manage.
        _model_versions: Tracked model hashes.

    Example:
        invalidator = CacheInvalidator([prompt_cache, compression_cache])
        invalidator.on_model_loaded("phi-4", "abc123")
        # Later, model reloaded with changes
        invalidator.on_model_loaded("phi-4", "def456")
        # Caches for phi-4 are invalidated
    """

    def __init__(self, caches: list[InferenceCache]) -> None:
        """Initialize CacheInvalidator.

        Args:
            caches: List of caches to manage invalidation for.
        """
        self.caches = caches
        self._model_versions: dict[str, str] = {}

    def on_model_loaded(self, model_id: str, model_hash: str) -> None:
        """Track model version on load, invalidate if changed.

        Args:
            model_id: Model identifier.
            model_hash: Hash of model file/config.
        """
        previous = self._model_versions.get(model_id)
        self._model_versions[model_id] = model_hash

        if previous and previous != model_hash:
            # Model changed - invalidate caches
            self._invalidate_for_model(model_id)

    def on_config_change(self, old_config: str, new_config: str) -> None:
        """Invalidate all caches on config change.

        Args:
            old_config: Previous config hash/content.
            new_config: New config hash/content.
        """
        if old_config != new_config:
            for cache in self.caches:
                cache.clear()

    def _invalidate_for_model(self, model_id: str) -> None:
        """Invalidate cache entries for specific model.

        Args:
            model_id: Model ID to invalidate.
        """
        for cache in self.caches:
            cache.invalidate_by_model(model_id)
