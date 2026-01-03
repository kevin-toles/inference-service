"""Unit tests for caching infrastructure.

Tests for WBS-INF16: Caching Infrastructure

AC-16.1: InferenceCache ABC with required methods
AC-16.2: PromptCache caches tokenized prompts
AC-16.3: HandoffCache with asyncio.Lock (AP-10.1)
AC-16.4: CompressionCache for compressed content
AC-16.5: CacheInvalidator invalidates on model change
"""

import asyncio
import hashlib
from abc import ABC
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# =============================================================================
# Helper Functions
# =============================================================================


def create_handoff_state(
    request_id: str = "req-default",
    goal: str = "Test goal",
    current_step: int = 1,
    total_steps: int = 3,
    **kwargs: Any,
) -> Any:
    """Create a HandoffState with required fields.

    Helper to create HandoffState instances with sensible defaults.
    """
    from src.orchestration.context import HandoffState

    return HandoffState(
        request_id=request_id,
        goal=goal,
        current_step=current_step,
        total_steps=total_steps,
        **kwargs,
    )


# =============================================================================
# Test: InferenceCache ABC (AC-16.1)
# =============================================================================


class TestInferenceCacheABC:
    """Tests for InferenceCache abstract base class."""

    def test_inference_cache_importable(self) -> None:
        """InferenceCache should be importable."""
        from src.services.cache import InferenceCache

        # Import succeeds = class exists and is importable
        assert InferenceCache  # Verify class is truthy (defined)

    def test_inference_cache_is_abc(self) -> None:
        """InferenceCache should be an abstract base class."""
        from src.services.cache import InferenceCache

        assert issubclass(InferenceCache, ABC), "InferenceCache must inherit from ABC"

    def test_inference_cache_not_instantiable(self) -> None:
        """InferenceCache should not be instantiable directly."""
        from src.services.cache import InferenceCache

        with pytest.raises(TypeError, match="abstract"):
            InferenceCache()  # type: ignore[abstract]

    def test_inference_cache_has_get_method(self) -> None:
        """InferenceCache should have abstract get method."""
        from src.services.cache import InferenceCache

        assert hasattr(InferenceCache, "get")
        # Check it's abstract
        assert getattr(InferenceCache.get, "__isabstractmethod__", False)

    def test_inference_cache_has_store_method(self) -> None:
        """InferenceCache should have abstract store method."""
        from src.services.cache import InferenceCache

        assert hasattr(InferenceCache, "store")
        assert getattr(InferenceCache.store, "__isabstractmethod__", False)

    def test_inference_cache_has_clear_method(self) -> None:
        """InferenceCache should have abstract clear method."""
        from src.services.cache import InferenceCache

        assert hasattr(InferenceCache, "clear")
        assert getattr(InferenceCache.clear, "__isabstractmethod__", False)

    def test_inference_cache_has_invalidate_by_model_method(self) -> None:
        """InferenceCache should have invalidate_by_model method."""
        from src.services.cache import InferenceCache

        assert hasattr(InferenceCache, "invalidate_by_model")
        assert getattr(
            InferenceCache.invalidate_by_model, "__isabstractmethod__", False
        )


class TestInferenceCacheImplementation:
    """Tests for verifying cache implementations."""

    def test_prompt_cache_implements_interface(self) -> None:
        """PromptCache should implement InferenceCache."""
        from src.services.cache import InferenceCache, PromptCache

        mock_tokenizer = MagicMock()
        cache = PromptCache(tokenizer=mock_tokenizer)
        assert isinstance(cache, InferenceCache)

    def test_handoff_cache_implements_interface(self) -> None:
        """HandoffCache should implement InferenceCache."""
        from src.services.cache import HandoffCache, InferenceCache

        cache = HandoffCache()
        assert isinstance(cache, InferenceCache)

    def test_compression_cache_implements_interface(self) -> None:
        """CompressionCache should implement InferenceCache."""
        from src.services.cache import CompressionCache, InferenceCache

        cache = CompressionCache()
        assert isinstance(cache, InferenceCache)


# =============================================================================
# Test: PromptCache (AC-16.2)
# =============================================================================


class TestPromptCacheImport:
    """Tests for PromptCache import and basic structure."""

    def test_prompt_cache_importable(self) -> None:
        """PromptCache should be importable."""
        from src.services.cache import PromptCache

        # Import succeeds = class exists and is importable
        assert PromptCache  # Verify class is truthy (defined)

    def test_prompt_cache_requires_tokenizer(self) -> None:
        """PromptCache should require a tokenizer."""
        from src.services.cache import PromptCache

        mock_tokenizer = MagicMock()
        cache = PromptCache(tokenizer=mock_tokenizer)
        # Verify cache instantiated successfully
        assert cache  # Instance is truthy


class TestPromptCacheTokenization:
    """Tests for PromptCache tokenization caching."""

    def test_get_tokenized_returns_tokens(self) -> None:
        """get_tokenized should return list of token IDs."""
        from src.services.cache import PromptCache

        mock_tokenizer = MagicMock()
        mock_tokenizer.tokenize.return_value = [1, 2, 3, 4, 5]

        cache = PromptCache(tokenizer=mock_tokenizer)
        result = cache.get_tokenized("Hello world")

        assert isinstance(result, list)
        assert result == [1, 2, 3, 4, 5]

    def test_get_tokenized_caches_result(self) -> None:
        """Same text should be tokenized only once."""
        from src.services.cache import PromptCache

        mock_tokenizer = MagicMock()
        mock_tokenizer.tokenize.return_value = [1, 2, 3]

        cache = PromptCache(tokenizer=mock_tokenizer)

        # Call twice with same text
        result1 = cache.get_tokenized("Hello world")
        result2 = cache.get_tokenized("Hello world")

        # Should only call tokenize once
        assert mock_tokenizer.tokenize.call_count == 1
        assert result1 == result2

    def test_get_tokenized_different_text_not_cached(self) -> None:
        """Different texts should be tokenized separately."""
        from src.services.cache import PromptCache

        mock_tokenizer = MagicMock()
        mock_tokenizer.tokenize.side_effect = [[1, 2, 3], [4, 5, 6]]

        cache = PromptCache(tokenizer=mock_tokenizer)

        result1 = cache.get_tokenized("Hello world")
        result2 = cache.get_tokenized("Goodbye world")

        assert mock_tokenizer.tokenize.call_count == 2
        assert result1 != result2

    def test_get_tokenized_uses_hash_key(self) -> None:
        """Cache should use hash of text as key."""
        from src.services.cache import PromptCache

        mock_tokenizer = MagicMock()
        mock_tokenizer.tokenize.return_value = [1, 2, 3]

        cache = PromptCache(tokenizer=mock_tokenizer)
        cache.get_tokenized("Test text")

        # Verify hash is used (internal detail)
        # MD5 is safe here - used only for cache key verification in tests
        expected_key = hashlib.md5("Test text".encode(), usedforsecurity=False).hexdigest()
        assert expected_key in cache._cache


class TestPromptCacheStore:
    """Tests for PromptCache store method."""

    def test_store_adds_to_cache(self) -> None:
        """store should add tokenized prompt to cache."""
        from src.services.cache import PromptCache

        mock_tokenizer = MagicMock()
        cache = PromptCache(tokenizer=mock_tokenizer)

        cache.store(key="test_key", value=[1, 2, 3])
        result = cache.get(key="test_key")

        assert result == [1, 2, 3]

    def test_get_returns_none_for_missing_key(self) -> None:
        """get should return None for missing keys."""
        from src.services.cache import PromptCache

        mock_tokenizer = MagicMock()
        cache = PromptCache(tokenizer=mock_tokenizer)

        result = cache.get(key="nonexistent")
        assert result is None


class TestPromptCacheClear:
    """Tests for PromptCache clear and invalidation."""

    def test_clear_empties_cache(self) -> None:
        """clear should empty the entire cache."""
        from src.services.cache import PromptCache

        mock_tokenizer = MagicMock()
        mock_tokenizer.tokenize.return_value = [1, 2, 3]

        cache = PromptCache(tokenizer=mock_tokenizer)
        cache.get_tokenized("Hello")
        cache.get_tokenized("World")

        cache.clear()

        # Cache should be empty, so tokenizer called again
        cache.get_tokenized("Hello")
        assert mock_tokenizer.tokenize.call_count == 3

    def test_invalidate_by_model_clears_all(self) -> None:
        """invalidate_by_model should clear cache for that model."""
        from src.services.cache import PromptCache

        mock_tokenizer = MagicMock()
        mock_tokenizer.tokenize.return_value = [1, 2, 3]

        cache = PromptCache(tokenizer=mock_tokenizer, model_id="phi-4")
        cache.get_tokenized("Hello")

        cache.invalidate_by_model("phi-4")

        # Should need to re-tokenize
        cache.get_tokenized("Hello")
        assert mock_tokenizer.tokenize.call_count == 2

    def test_invalidate_by_model_ignores_other_models(self) -> None:
        """invalidate_by_model should not affect other models."""
        from src.services.cache import PromptCache

        mock_tokenizer = MagicMock()
        mock_tokenizer.tokenize.return_value = [1, 2, 3]

        cache = PromptCache(tokenizer=mock_tokenizer, model_id="phi-4")
        cache.get_tokenized("Hello")

        cache.invalidate_by_model("deepseek-r1-7b")

        # Should still be cached
        cache.get_tokenized("Hello")
        assert mock_tokenizer.tokenize.call_count == 1


# =============================================================================
# Test: HandoffCache (AC-16.3)
# =============================================================================


class TestHandoffCacheImport:
    """Tests for HandoffCache import and structure."""

    def test_handoff_cache_importable(self) -> None:
        """HandoffCache should be importable."""
        from src.services.cache import HandoffCache

        # Import succeeds = class exists and is importable
        assert HandoffCache  # Verify class is truthy (defined)

    def test_handoff_cache_instantiates_without_redis(self) -> None:
        """HandoffCache should work without Redis (local fallback)."""
        from src.services.cache import HandoffCache

        cache = HandoffCache()
        # Verify cache instantiated successfully
        assert cache  # Instance is truthy

    def test_handoff_cache_accepts_redis_client(self) -> None:
        """HandoffCache should accept optional Redis client."""
        from src.services.cache import HandoffCache

        mock_redis = MagicMock()
        cache = HandoffCache(redis_client=mock_redis)
        assert cache._redis == mock_redis


class TestHandoffCacheLocking:
    """Tests for HandoffCache asyncio.Lock (AP-10.1)."""

    def test_handoff_cache_has_locks_dict(self) -> None:
        """HandoffCache should have per-resource locks dict."""
        from src.services.cache import HandoffCache

        cache = HandoffCache()
        assert hasattr(cache, "_locks")
        assert isinstance(cache._locks, dict)

    def test_get_lock_creates_lock_for_key(self) -> None:
        """_get_lock should create asyncio.Lock for new keys."""
        from src.services.cache import HandoffCache

        cache = HandoffCache()
        lock = cache._get_lock("test_key")

        assert isinstance(lock, asyncio.Lock)
        assert "test_key" in cache._locks

    def test_get_lock_returns_same_lock_for_same_key(self) -> None:
        """_get_lock should return same lock for same key."""
        from src.services.cache import HandoffCache

        cache = HandoffCache()
        lock1 = cache._get_lock("key1")
        lock2 = cache._get_lock("key1")

        assert lock1 is lock2

    def test_get_lock_returns_different_lock_for_different_keys(self) -> None:
        """_get_lock should return different locks for different keys."""
        from src.services.cache import HandoffCache

        cache = HandoffCache()
        lock1 = cache._get_lock("key1")
        lock2 = cache._get_lock("key2")

        assert lock1 is not lock2


class TestHandoffCacheStore:
    """Tests for HandoffCache store method."""

    async def test_store_saves_handoff_state_locally(self) -> None:
        """store should save HandoffState to local dict without Redis."""
        from src.services.cache import HandoffCache

        cache = HandoffCache()
        state = create_handoff_state(request_id="req-123")

        await cache.store(state)

        assert "handoff:req-123" in cache._local

    async def test_store_uses_lock(self) -> None:
        """store should acquire lock before writing."""
        from src.services.cache import HandoffCache

        cache = HandoffCache()
        state = create_handoff_state(request_id="req-123")

        # Pre-create lock to verify it's used
        lock = cache._get_lock("handoff:req-123")

        async with lock:
            # Lock is held, store should wait
            task = asyncio.create_task(cache.store(state))
            await asyncio.sleep(0.01)
            assert not task.done()

        # Now it should complete
        await task

    async def test_store_with_ttl(self) -> None:
        """store should accept TTL parameter."""
        from src.services.cache import HandoffCache

        cache = HandoffCache()
        state = create_handoff_state(request_id="req-123")

        await cache.store(state, ttl=300)

        # TTL not enforced in local mode, but should accept param
        assert "handoff:req-123" in cache._local


class TestHandoffCacheGet:
    """Tests for HandoffCache get method."""

    async def test_get_retrieves_stored_state(self) -> None:
        """get should retrieve previously stored HandoffState."""
        from src.services.cache import HandoffCache

        cache = HandoffCache()
        state = create_handoff_state(
            request_id="req-456",
            decisions_made=["decision1"],
        )

        await cache.store(state)
        retrieved = await cache.get("req-456")

        assert retrieved is not None
        assert retrieved.request_id == "req-456"
        assert retrieved.decisions_made == ["decision1"]

    async def test_get_returns_none_for_missing(self) -> None:
        """get should return None for non-existent request_id."""
        from src.services.cache import HandoffCache

        cache = HandoffCache()
        result = await cache.get("nonexistent")

        assert result is None

    async def test_get_uses_lock(self) -> None:
        """get should acquire lock before reading."""
        from src.services.cache import HandoffCache

        cache = HandoffCache()
        state = create_handoff_state(request_id="req-789")
        await cache.store(state)

        lock = cache._get_lock("handoff:req-789")

        async with lock:
            task = asyncio.create_task(cache.get("req-789"))
            await asyncio.sleep(0.01)
            assert not task.done()

        result = await task
        assert result is not None


class TestHandoffCacheConcurrency:
    """Tests for HandoffCache race condition prevention."""

    async def test_concurrent_writes_dont_lose_data(self) -> None:
        """Concurrent writes to different keys should not interfere."""
        from src.services.cache import HandoffCache

        cache = HandoffCache()

        async def write_state(request_id: str) -> None:
            state = create_handoff_state(request_id=request_id)
            await cache.store(state)

        # Concurrent writes to different keys
        await asyncio.gather(
            write_state("req-001"),
            write_state("req-002"),
            write_state("req-003"),
        )

        # All should be stored
        assert await cache.get("req-001") is not None
        assert await cache.get("req-002") is not None
        assert await cache.get("req-003") is not None

    async def test_concurrent_read_write_same_key(self) -> None:
        """Concurrent read/write to same key should be serialized."""
        from src.services.cache import HandoffCache

        cache = HandoffCache()
        state = create_handoff_state(request_id="req-concurrent")
        await cache.store(state)

        results: list[Any] = []

        async def read_and_append() -> None:
            result = await cache.get("req-concurrent")
            results.append(result)

        async def write_updated() -> None:
            new_state = create_handoff_state(
                request_id="req-concurrent",
                decisions_made=["updated"],
            )
            await cache.store(new_state)

        # Mix reads and writes
        await asyncio.gather(
            read_and_append(),
            write_updated(),
            read_and_append(),
        )

        # All reads should complete
        assert len(results) == 2


class TestHandoffCacheClear:
    """Tests for HandoffCache clear and invalidation."""

    async def test_clear_empties_local_cache(self) -> None:
        """clear should empty local cache."""
        from src.services.cache import HandoffCache

        cache = HandoffCache()
        await cache.store(create_handoff_state(request_id="req-1"))
        await cache.store(create_handoff_state(request_id="req-2"))

        cache.clear()

        assert await cache.get("req-1") is None
        assert await cache.get("req-2") is None

    def test_invalidate_by_model_clears_all(self) -> None:
        """invalidate_by_model should clear all handoff states."""
        from src.services.cache import HandoffCache

        cache = HandoffCache()
        cache._local["handoff:req-1"] = "{}"
        cache._local["handoff:req-2"] = "{}"

        cache.invalidate_by_model("any-model")

        assert len(cache._local) == 0


# =============================================================================
# Test: CompressionCache (AC-16.4)
# =============================================================================


class TestCompressionCacheImport:
    """Tests for CompressionCache import and structure."""

    def test_compression_cache_importable(self) -> None:
        """CompressionCache should be importable."""
        from src.services.cache import CompressionCache

        # Import succeeds = class exists and is importable
        assert CompressionCache  # Verify class is truthy (defined)

    def test_compression_cache_instantiates(self) -> None:
        """CompressionCache should instantiate without arguments."""
        from src.services.cache import CompressionCache

        cache = CompressionCache()
        # Verify cache instantiated successfully
        assert cache  # Instance is truthy


class TestCompressionCacheKeyGeneration:
    """Tests for CompressionCache key generation."""

    def test_get_key_returns_string(self) -> None:
        """get_key should return string key."""
        from src.services.cache import CompressionCache

        cache = CompressionCache()
        key = cache.get_key("test content", 100)

        assert isinstance(key, str)

    def test_get_key_includes_target_tokens(self) -> None:
        """Key should include target_tokens for uniqueness."""
        from src.services.cache import CompressionCache

        cache = CompressionCache()
        key1 = cache.get_key("test content", 100)
        key2 = cache.get_key("test content", 200)

        assert key1 != key2
        assert "100" in key1
        assert "200" in key2

    def test_get_key_uses_content_hash(self) -> None:
        """Key should use hash of content."""
        from src.services.cache import CompressionCache

        cache = CompressionCache()
        key1 = cache.get_key("content A", 100)
        key2 = cache.get_key("content B", 100)

        assert key1 != key2

    def test_get_key_same_content_same_tokens(self) -> None:
        """Same content and tokens should produce same key."""
        from src.services.cache import CompressionCache

        cache = CompressionCache()
        key1 = cache.get_key("same content", 100)
        key2 = cache.get_key("same content", 100)

        assert key1 == key2


class TestCompressionCacheStore:
    """Tests for CompressionCache store method."""

    def test_store_saves_compressed_content(self) -> None:
        """store should save compressed content."""
        from src.services.cache import CompressionCache

        cache = CompressionCache()
        cache.store("original content", 100, "compressed content")

        result = cache.get("original content", 100)
        assert result == "compressed content"

    def test_store_different_target_tokens(self) -> None:
        """Same content compressed to different sizes stored separately."""
        from src.services.cache import CompressionCache

        cache = CompressionCache()
        cache.store("original", 100, "compressed to 100")
        cache.store("original", 200, "compressed to 200")

        assert cache.get("original", 100) == "compressed to 100"
        assert cache.get("original", 200) == "compressed to 200"


class TestCompressionCacheGet:
    """Tests for CompressionCache get method."""

    def test_get_returns_cached_compression(self) -> None:
        """get should return cached compression."""
        from src.services.cache import CompressionCache

        cache = CompressionCache()
        cache.store("test", 50, "compressed test")

        result = cache.get("test", 50)
        assert result == "compressed test"

    def test_get_returns_none_for_missing(self) -> None:
        """get should return None for missing entries."""
        from src.services.cache import CompressionCache

        cache = CompressionCache()
        result = cache.get("nonexistent", 100)

        assert result is None

    def test_get_with_key_returns_by_key(self) -> None:
        """get with key parameter should lookup by key."""
        from src.services.cache import CompressionCache

        cache = CompressionCache()
        cache.store(key="direct_key", value="direct_value")

        result = cache.get(key="direct_key")
        assert result == "direct_value"


class TestCompressionCacheClear:
    """Tests for CompressionCache clear and invalidation."""

    def test_clear_empties_cache(self) -> None:
        """clear should empty entire cache."""
        from src.services.cache import CompressionCache

        cache = CompressionCache()
        cache.store("a", 100, "compressed a")
        cache.store("b", 200, "compressed b")

        cache.clear()

        assert cache.get("a", 100) is None
        assert cache.get("b", 200) is None

    def test_invalidate_by_model_clears_all(self) -> None:
        """invalidate_by_model should clear all entries."""
        from src.services.cache import CompressionCache

        cache = CompressionCache()
        cache.store("content", 100, "compressed")

        cache.invalidate_by_model("any-model")

        assert cache.get("content", 100) is None


# =============================================================================
# Test: CacheInvalidator (AC-16.5)
# =============================================================================


class TestCacheInvalidatorImport:
    """Tests for CacheInvalidator import and structure."""

    def test_cache_invalidator_importable(self) -> None:
        """CacheInvalidator should be importable."""
        from src.services.cache import CacheInvalidator

        # Import succeeds = class exists and is importable
        assert CacheInvalidator  # Verify class is truthy (defined)

    def test_cache_invalidator_accepts_cache_list(self) -> None:
        """CacheInvalidator should accept list of caches."""
        from src.services.cache import (
            CacheInvalidator,
            CompressionCache,
            PromptCache,
        )

        mock_tokenizer = MagicMock()
        caches = [
            PromptCache(tokenizer=mock_tokenizer),
            CompressionCache(),
        ]
        invalidator = CacheInvalidator(caches=caches)

        # Verify invalidator instantiated successfully
        assert invalidator  # Instance is truthy
        assert len(invalidator.caches) == 2


class TestCacheInvalidatorModelTracking:
    """Tests for CacheInvalidator model version tracking."""

    def test_on_model_loaded_tracks_version(self) -> None:
        """on_model_loaded should track model hash."""
        from src.services.cache import CacheInvalidator

        invalidator = CacheInvalidator(caches=[])

        invalidator.on_model_loaded("phi-4", "hash123")

        assert invalidator._model_versions.get("phi-4") == "hash123"

    def test_on_model_loaded_updates_version(self) -> None:
        """on_model_loaded should update version on change."""
        from src.services.cache import CacheInvalidator

        invalidator = CacheInvalidator(caches=[])

        invalidator.on_model_loaded("phi-4", "hash123")
        invalidator.on_model_loaded("phi-4", "hash456")

        assert invalidator._model_versions.get("phi-4") == "hash456"


class TestCacheInvalidatorInvalidation:
    """Tests for CacheInvalidator invalidation logic."""

    def test_on_model_loaded_invalidates_on_hash_change(self) -> None:
        """Model hash change should invalidate caches."""
        from src.services.cache import CacheInvalidator, CompressionCache

        cache = CompressionCache()
        cache.store("content", 100, "compressed")
        invalidator = CacheInvalidator(caches=[cache])

        # First load
        invalidator.on_model_loaded("phi-4", "hash-v1")

        # Content should still be cached
        assert cache.get("content", 100) == "compressed"

        # Second load with different hash
        cache.store("content", 100, "compressed")  # Re-add
        invalidator.on_model_loaded("phi-4", "hash-v2")

        # Cache should be invalidated
        assert cache.get("content", 100) is None

    def test_on_model_loaded_no_invalidation_same_hash(self) -> None:
        """Same model hash should not invalidate caches."""
        from src.services.cache import CacheInvalidator, CompressionCache

        cache = CompressionCache()
        cache.store("content", 100, "compressed")
        invalidator = CacheInvalidator(caches=[cache])

        invalidator.on_model_loaded("phi-4", "hash-v1")
        invalidator.on_model_loaded("phi-4", "hash-v1")  # Same hash

        # Should still be cached
        assert cache.get("content", 100) == "compressed"

    def test_on_config_change_invalidates_all(self) -> None:
        """Config change should invalidate all caches."""
        from src.services.cache import CacheInvalidator, CompressionCache

        cache1 = CompressionCache()
        cache2 = CompressionCache()
        cache1.store("a", 100, "compressed a")
        cache2.store("b", 200, "compressed b")

        invalidator = CacheInvalidator(caches=[cache1, cache2])

        invalidator.on_config_change("old_config", "new_config")

        assert cache1.get("a", 100) is None
        assert cache2.get("b", 200) is None

    def test_on_config_change_no_invalidation_same_config(self) -> None:
        """Same config should not invalidate caches."""
        from src.services.cache import CacheInvalidator, CompressionCache

        cache = CompressionCache()
        cache.store("content", 100, "compressed")
        invalidator = CacheInvalidator(caches=[cache])

        invalidator.on_config_change("same_config", "same_config")

        assert cache.get("content", 100) == "compressed"

    def test_invalidate_for_model_calls_all_caches(self) -> None:
        """_invalidate_for_model should call invalidate_by_model on all caches."""
        from src.services.cache import CacheInvalidator, InferenceCache

        mock_cache1 = MagicMock(spec=InferenceCache)
        mock_cache2 = MagicMock(spec=InferenceCache)

        invalidator = CacheInvalidator(caches=[mock_cache1, mock_cache2])
        invalidator._invalidate_for_model("phi-4")

        mock_cache1.invalidate_by_model.assert_called_once_with("phi-4")
        mock_cache2.invalidate_by_model.assert_called_once_with("phi-4")


class TestCacheInvalidatorEdgeCases:
    """Tests for CacheInvalidator edge cases."""

    def test_empty_cache_list(self) -> None:
        """Invalidator should work with empty cache list."""
        from src.services.cache import CacheInvalidator

        invalidator = CacheInvalidator(caches=[])

        # Should not raise
        invalidator.on_model_loaded("phi-4", "hash")
        invalidator.on_config_change("old", "new")

    def test_first_model_load_no_invalidation(self) -> None:
        """First model load should not invalidate (no previous version)."""
        from src.services.cache import CacheInvalidator, CompressionCache

        cache = CompressionCache()
        cache.store("content", 100, "compressed")
        invalidator = CacheInvalidator(caches=[cache])

        # First load - no previous version
        invalidator.on_model_loaded("new-model", "hash-first")

        # Should still be cached
        assert cache.get("content", 100) == "compressed"


# =============================================================================
# Test: Integration
# =============================================================================


class TestCacheIntegration:
    """Integration tests for cache system."""

    async def test_all_caches_work_together(self) -> None:
        """All caches should work with CacheInvalidator."""
        from src.services.cache import (
            CacheInvalidator,
            CompressionCache,
            HandoffCache,
            PromptCache,
        )

        mock_tokenizer = MagicMock()
        mock_tokenizer.tokenize.return_value = [1, 2, 3]

        prompt_cache = PromptCache(tokenizer=mock_tokenizer, model_id="phi-4")
        handoff_cache = HandoffCache()
        compression_cache = CompressionCache()

        invalidator = CacheInvalidator(
            caches=[prompt_cache, handoff_cache, compression_cache]
        )

        # Populate caches
        prompt_cache.get_tokenized("System prompt")
        await handoff_cache.store(create_handoff_state(request_id="req-int"))
        compression_cache.store("long content", 100, "short")

        # Invalidate on model change
        invalidator.on_model_loaded("phi-4", "v1")
        invalidator.on_model_loaded("phi-4", "v2")

        # All should be cleared
        assert compression_cache.get("long content", 100) is None

    def test_cache_key_collision_prevented(self) -> None:
        """Different content should not collide in cache."""
        from src.services.cache import CompressionCache

        cache = CompressionCache()

        # Store many entries
        for i in range(100):
            cache.store(f"content-{i}", 100, f"compressed-{i}")

        # All should be retrievable
        for i in range(100):
            assert cache.get(f"content-{i}", 100) == f"compressed-{i}"
