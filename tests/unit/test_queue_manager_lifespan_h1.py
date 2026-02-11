"""Unit tests for H-1: QueueManager wired into application lifespan.

Tests cover:
- AC-H1.1: app.state.queue_manager is available after startup
- AC-H1.2: max_concurrent=1 for llama-cpp (single GPU, single request at a time)
- AC-H1.3: Clean shutdown awaits pending requests before exit

Also tests the new QueueManager.shutdown() method.
"""

import asyncio

import pytest
from fastapi.testclient import TestClient

from src.services.queue_manager import QueueManager, QueueStrategy, RequestItem


# =============================================================================
# QueueManager.shutdown() unit tests
# =============================================================================


class TestQueueManagerShutdown:
    """Test QueueManager.shutdown() method."""

    @pytest.mark.asyncio
    async def test_shutdown_returns_stats(self) -> None:
        """shutdown() returns dict with active_drained, pending_cleared, timed_out."""
        qm = QueueManager(max_concurrent=1, strategy=QueueStrategy.FIFO)
        result = await qm.shutdown()
        assert result == {
            "active_drained": 0,
            "pending_cleared": 0,
            "timed_out": False,
        }

    @pytest.mark.asyncio
    async def test_shutdown_clears_pending_items(self) -> None:
        """shutdown() clears all pending items from the queue."""
        qm = QueueManager(max_concurrent=1, strategy=QueueStrategy.FIFO)
        # Enqueue some items
        for i in range(3):
            await qm.enqueue(RequestItem(request_id=f"req-{i}", data={"prompt": "test"}))
        assert qm.pending_count == 3

        result = await qm.shutdown()
        assert result["pending_cleared"] == 3
        assert qm.is_empty

    @pytest.mark.asyncio
    async def test_shutdown_waits_for_active_to_drain(self) -> None:
        """shutdown() waits for active requests to complete before returning."""
        qm = QueueManager(max_concurrent=1, strategy=QueueStrategy.FIFO)
        await qm.acquire_slot()
        assert qm.active_count == 1

        # Simulate a request finishing after 200ms
        async def release_after_delay() -> None:
            await asyncio.sleep(0.2)
            qm.release_slot()

        asyncio.create_task(release_after_delay())

        result = await qm.shutdown(timeout=5.0)
        assert result["active_drained"] == 1
        assert result["timed_out"] is False
        assert qm.active_count == 0

    @pytest.mark.asyncio
    async def test_shutdown_times_out_if_active_stuck(self) -> None:
        """shutdown() returns timed_out=True if active requests don't complete."""
        qm = QueueManager(max_concurrent=1, strategy=QueueStrategy.FIFO)
        await qm.acquire_slot()
        assert qm.active_count == 1

        # Don't release the slot — shutdown should time out
        result = await qm.shutdown(timeout=0.3)
        assert result["timed_out"] is True
        # The slot is still held
        assert qm.active_count == 1

    @pytest.mark.asyncio
    async def test_shutdown_drains_active_and_clears_pending(self) -> None:
        """shutdown() handles both active draining and pending clearing."""
        qm = QueueManager(max_concurrent=2, strategy=QueueStrategy.FIFO)
        await qm.acquire_slot()
        await qm.enqueue(RequestItem(request_id="pending-1", data={"prompt": "test"}))
        await qm.enqueue(RequestItem(request_id="pending-2", data={"prompt": "test"}))

        # Release the active slot quickly
        async def release_after_delay() -> None:
            await asyncio.sleep(0.1)
            qm.release_slot()

        asyncio.create_task(release_after_delay())

        result = await qm.shutdown(timeout=5.0)
        assert result["active_drained"] == 1
        assert result["pending_cleared"] == 2
        assert result["timed_out"] is False


# =============================================================================
# Lifespan integration tests — QueueManager wired into app
# =============================================================================


class TestQueueManagerLifespan:
    """Test QueueManager initialization and shutdown in FastAPI lifespan."""

    def test_queue_manager_available_after_startup(self) -> None:
        """AC-H1.1: app.state.queue_manager is available after startup."""
        from src.main import app

        client = TestClient(app)
        with client:
            assert hasattr(app.state, "queue_manager")
            assert app.state.queue_manager is not None
            assert isinstance(app.state.queue_manager, QueueManager)

    def test_queue_manager_max_concurrent_is_one(self) -> None:
        """AC-H1.2: max_concurrent=1 for single GPU serialization."""
        from src.main import app

        client = TestClient(app)
        with client:
            qm = app.state.queue_manager
            assert qm.max_concurrent == 1

    def test_queue_manager_uses_fifo_strategy(self) -> None:
        """QueueManager uses FIFO strategy for fair request ordering."""
        from src.main import app

        client = TestClient(app)
        with client:
            qm = app.state.queue_manager
            assert qm._strategy == QueueStrategy.FIFO

    def test_queue_manager_starts_empty(self) -> None:
        """QueueManager starts with zero active and zero pending."""
        from src.main import app

        client = TestClient(app)
        with client:
            qm = app.state.queue_manager
            assert qm.active_count == 0
            assert qm.pending_count == 0
            assert qm.is_empty

    def test_lifespan_shutdown_cleans_up_queue_manager(self) -> None:
        """AC-H1.3: Clean shutdown via lifespan (TestClient triggers it)."""
        from src.main import app

        client = TestClient(app)
        with client:
            qm = app.state.queue_manager
            assert qm is not None
        # After exiting context, lifespan shutdown has run.
        # Verify app.state.initialized is False (existing shutdown behavior).
        assert app.state.initialized is False


class TestQueueManagerLifespanCoexistence:
    """Ensure QueueManager doesn't break existing lifespan functionality."""

    def test_health_endpoint_still_works(self) -> None:
        """Health endpoint continues to function with QueueManager in lifespan."""
        from src.main import app

        client = TestClient(app)
        with client:
            response = client.get("/health")
            assert response.status_code == 200
            assert response.json()["status"] == "ok"

    def test_model_manager_still_initialized(self) -> None:
        """model_manager is still available alongside queue_manager."""
        from src.main import app

        client = TestClient(app)
        with client:
            assert hasattr(app.state, "model_manager")
            assert app.state.model_manager is not None
            assert hasattr(app.state, "queue_manager")
            assert app.state.queue_manager is not None
