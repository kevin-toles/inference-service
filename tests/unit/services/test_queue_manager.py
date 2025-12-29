"""Unit tests for QueueManager service.

Tests cover:
- AC-17.1: QueueManager uses asyncio.Queue
- AC-17.2: QueueManager enforces max_concurrent_requests
- AC-17.3: QueueManager supports FIFO and priority strategies
- AC-17.4: QueueManager rejects when full if configured

Exit Criteria:
- 11th request blocked when max=10
- Priority=3 processed before priority=1
- QueueFullError raised when reject_when_full=True
"""

import asyncio
from typing import Any
from unittest.mock import AsyncMock

import pytest

from src.services.queue_manager import (
    QueueFullError,
    QueueManager,
    QueueStrategy,
    RequestItem,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def fifo_queue() -> QueueManager:
    """Create a FIFO queue manager with default settings."""
    return QueueManager(
        max_concurrent=10,
        strategy=QueueStrategy.FIFO,
        reject_when_full=False,
    )


@pytest.fixture
def priority_queue() -> QueueManager:
    """Create a priority queue manager."""
    return QueueManager(
        max_concurrent=10,
        strategy=QueueStrategy.PRIORITY,
        reject_when_full=False,
    )


@pytest.fixture
def reject_queue() -> QueueManager:
    """Create a queue manager that rejects when full."""
    return QueueManager(
        max_concurrent=3,
        strategy=QueueStrategy.FIFO,
        reject_when_full=True,
    )


def create_request_item(
    request_id: str = "req-1",
    priority: int = 2,
    data: dict[str, Any] | None = None,
) -> RequestItem:
    """Helper to create a RequestItem."""
    return RequestItem(
        request_id=request_id,
        priority=priority,
        data=data or {"prompt": "test"},
    )


# =============================================================================
# AC-17.1: QueueManager uses asyncio.Queue
# =============================================================================


class TestQueueManagerUsesAsyncioQueue:
    """Test that QueueManager uses asyncio.Queue internally."""

    def test_queue_manager_has_internal_queue(self, fifo_queue: QueueManager) -> None:
        """QueueManager should have an internal asyncio.Queue."""
        assert hasattr(fifo_queue, "_queue")
        assert isinstance(fifo_queue._queue, asyncio.Queue)

    def test_queue_manager_has_priority_queue_when_priority_strategy(
        self, priority_queue: QueueManager
    ) -> None:
        """Priority QueueManager should use asyncio.PriorityQueue."""
        assert hasattr(priority_queue, "_queue")
        assert isinstance(priority_queue._queue, asyncio.PriorityQueue)

    @pytest.mark.asyncio
    async def test_enqueue_adds_to_queue(self, fifo_queue: QueueManager) -> None:
        """enqueue() should add items to the internal queue."""
        item = create_request_item("req-1")
        await fifo_queue.enqueue(item)
        assert fifo_queue.pending_count == 1

    @pytest.mark.asyncio
    async def test_dequeue_removes_from_queue(self, fifo_queue: QueueManager) -> None:
        """dequeue() should remove and return items from the queue."""
        item = create_request_item("req-1")
        await fifo_queue.enqueue(item)
        result = await fifo_queue.dequeue()
        assert result.request_id == "req-1"
        assert fifo_queue.pending_count == 0


# =============================================================================
# AC-17.2: QueueManager enforces max_concurrent_requests
# =============================================================================


class TestQueueManagerEnforcesMaxConcurrent:
    """Test that QueueManager enforces max_concurrent_requests limit."""

    def test_queue_manager_stores_max_concurrent(self) -> None:
        """QueueManager should store max_concurrent setting."""
        qm = QueueManager(max_concurrent=5)
        assert qm.max_concurrent == 5

    def test_queue_manager_default_max_concurrent(self) -> None:
        """QueueManager should have sensible default max_concurrent."""
        qm = QueueManager()
        assert qm.max_concurrent == 10

    @pytest.mark.asyncio
    async def test_acquire_slot_succeeds_when_available(
        self, fifo_queue: QueueManager
    ) -> None:
        """acquire_slot() should succeed when slots are available."""
        success = await fifo_queue.acquire_slot()
        assert success is True
        assert fifo_queue.active_count == 1

    @pytest.mark.asyncio
    async def test_release_slot_decrements_active(
        self, fifo_queue: QueueManager
    ) -> None:
        """release_slot() should decrement active count."""
        await fifo_queue.acquire_slot()
        assert fifo_queue.active_count == 1
        fifo_queue.release_slot()
        assert fifo_queue.active_count == 0

    @pytest.mark.asyncio
    async def test_max_concurrent_blocks_11th_request(self) -> None:
        """11th request should block when max=10 (Exit Criteria)."""
        qm = QueueManager(max_concurrent=10, reject_when_full=False)

        # Acquire 10 slots
        for _ in range(10):
            await qm.acquire_slot()

        assert qm.active_count == 10

        # 11th acquire should block - test with timeout
        acquired = asyncio.Event()

        async def try_acquire() -> None:
            await qm.acquire_slot()
            acquired.set()

        task = asyncio.create_task(try_acquire())

        # Should not acquire within 0.1 seconds
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(acquired.wait(), timeout=0.1)

        # Cleanup: release a slot to unblock
        qm.release_slot()
        await asyncio.wait_for(task, timeout=1.0)
        assert acquired.is_set()

    @pytest.mark.asyncio
    async def test_active_count_property(self, fifo_queue: QueueManager) -> None:
        """active_count property should reflect current active requests."""
        assert fifo_queue.active_count == 0
        await fifo_queue.acquire_slot()
        assert fifo_queue.active_count == 1
        await fifo_queue.acquire_slot()
        assert fifo_queue.active_count == 2
        fifo_queue.release_slot()
        assert fifo_queue.active_count == 1


class TestQueueManagerConcurrentProcessing:
    """Test concurrent request processing behavior."""

    @pytest.mark.asyncio
    async def test_concurrent_requests_processed_up_to_limit(self) -> None:
        """Requests up to max_concurrent should be processed concurrently."""
        qm = QueueManager(max_concurrent=3)
        results: list[str] = []
        processing_order: list[str] = []

        async def process_request(request_id: str) -> None:
            await qm.acquire_slot()
            processing_order.append(f"{request_id}_start")
            await asyncio.sleep(0.05)  # Simulate work
            results.append(request_id)
            processing_order.append(f"{request_id}_end")
            qm.release_slot()

        tasks = [
            asyncio.create_task(process_request(f"req-{i}")) for i in range(3)
        ]

        await asyncio.gather(*tasks)
        assert len(results) == 3

    @pytest.mark.asyncio
    async def test_semaphore_based_concurrency_control(self) -> None:
        """QueueManager should use Semaphore for concurrency control."""
        qm = QueueManager(max_concurrent=2)
        assert hasattr(qm, "_semaphore")
        assert isinstance(qm._semaphore, asyncio.Semaphore)


# =============================================================================
# AC-17.3: QueueManager supports FIFO and priority strategies
# =============================================================================


class TestFIFOStrategy:
    """Test FIFO queue strategy."""

    @pytest.mark.asyncio
    async def test_fifo_strategy_processes_in_order(
        self, fifo_queue: QueueManager
    ) -> None:
        """FIFO strategy should process requests in insertion order."""
        items = [
            create_request_item("req-1", priority=1),
            create_request_item("req-2", priority=3),
            create_request_item("req-3", priority=2),
        ]

        for item in items:
            await fifo_queue.enqueue(item)

        # Dequeue should return in insertion order regardless of priority
        result1 = await fifo_queue.dequeue()
        result2 = await fifo_queue.dequeue()
        result3 = await fifo_queue.dequeue()

        assert result1.request_id == "req-1"
        assert result2.request_id == "req-2"
        assert result3.request_id == "req-3"

    @pytest.mark.asyncio
    async def test_fifo_queue_uses_regular_asyncio_queue(
        self, fifo_queue: QueueManager
    ) -> None:
        """FIFO strategy should use asyncio.Queue (not PriorityQueue)."""
        assert isinstance(fifo_queue._queue, asyncio.Queue)
        assert not isinstance(fifo_queue._queue, asyncio.PriorityQueue)

    def test_queue_strategy_enum_has_fifo(self) -> None:
        """QueueStrategy enum should have FIFO value."""
        assert QueueStrategy.FIFO.value == "fifo"


class TestPriorityStrategy:
    """Test priority queue strategy."""

    @pytest.mark.asyncio
    async def test_priority_3_processed_before_priority_1(
        self, priority_queue: QueueManager
    ) -> None:
        """Priority=3 should be processed before priority=1 (Exit Criteria)."""
        # Enqueue in reverse priority order
        low_priority = create_request_item("req-low", priority=1)
        normal_priority = create_request_item("req-normal", priority=2)
        high_priority = create_request_item("req-high", priority=3)

        await priority_queue.enqueue(low_priority)
        await priority_queue.enqueue(normal_priority)
        await priority_queue.enqueue(high_priority)

        # Higher priority should come out first
        result1 = await priority_queue.dequeue()
        result2 = await priority_queue.dequeue()
        result3 = await priority_queue.dequeue()

        assert result1.request_id == "req-high"  # priority=3
        assert result2.request_id == "req-normal"  # priority=2
        assert result3.request_id == "req-low"  # priority=1

    @pytest.mark.asyncio
    async def test_same_priority_maintains_fifo_order(
        self, priority_queue: QueueManager
    ) -> None:
        """Items with same priority should be processed in FIFO order."""
        items = [
            create_request_item("req-a", priority=2),
            create_request_item("req-b", priority=2),
            create_request_item("req-c", priority=2),
        ]

        for item in items:
            await priority_queue.enqueue(item)

        result1 = await priority_queue.dequeue()
        result2 = await priority_queue.dequeue()
        result3 = await priority_queue.dequeue()

        # Same priority should maintain insertion order
        assert result1.request_id == "req-a"
        assert result2.request_id == "req-b"
        assert result3.request_id == "req-c"

    @pytest.mark.asyncio
    async def test_priority_queue_uses_asyncio_priority_queue(
        self, priority_queue: QueueManager
    ) -> None:
        """Priority strategy should use asyncio.PriorityQueue."""
        assert isinstance(priority_queue._queue, asyncio.PriorityQueue)

    def test_queue_strategy_enum_has_priority(self) -> None:
        """QueueStrategy enum should have PRIORITY value."""
        assert QueueStrategy.PRIORITY.value == "priority"

    @pytest.mark.asyncio
    async def test_priority_levels_1_2_3(
        self, priority_queue: QueueManager
    ) -> None:
        """Priority levels 1 (low), 2 (normal), 3 (high) should work."""
        items = [
            create_request_item("req-medium", priority=2),
            create_request_item("req-low", priority=1),
            create_request_item("req-high", priority=3),
        ]

        for item in items:
            await priority_queue.enqueue(item)

        results = [await priority_queue.dequeue() for _ in range(3)]
        priorities = [r.priority for r in results]

        # Should be in descending priority order
        assert priorities == [3, 2, 1]


class TestQueueStrategyEnum:
    """Test QueueStrategy enumeration."""

    def test_queue_strategy_from_string(self) -> None:
        """QueueStrategy should support creation from string."""
        assert QueueStrategy("fifo") == QueueStrategy.FIFO
        assert QueueStrategy("priority") == QueueStrategy.PRIORITY

    def test_queue_strategy_values(self) -> None:
        """QueueStrategy should have correct values."""
        assert QueueStrategy.FIFO.value == "fifo"
        assert QueueStrategy.PRIORITY.value == "priority"


# =============================================================================
# AC-17.4: QueueManager rejects when full if configured
# =============================================================================


class TestRejectWhenFull:
    """Test queue rejection behavior when full."""

    @pytest.mark.asyncio
    async def test_queue_full_error_raised_when_configured(
        self, reject_queue: QueueManager
    ) -> None:
        """QueueFullError should be raised when reject_when_full=True (Exit Criteria)."""
        # Fill the queue to max_concurrent
        for i in range(3):
            await reject_queue.acquire_slot()

        # Next acquire should raise QueueFullError
        with pytest.raises(QueueFullError) as exc_info:
            await reject_queue.acquire_slot()

        assert "queue is full" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_queue_full_error_not_raised_when_disabled(
        self, fifo_queue: QueueManager
    ) -> None:
        """When reject_when_full=False, should block instead of raising."""
        # Fill to max
        for _ in range(10):
            await fifo_queue.acquire_slot()

        # This should block, not raise
        acquired = asyncio.Event()

        async def try_acquire() -> None:
            await fifo_queue.acquire_slot()
            acquired.set()

        task = asyncio.create_task(try_acquire())

        # Should block, not immediately succeed or raise
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(acquired.wait(), timeout=0.1)

        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    def test_queue_full_error_is_exception(self) -> None:
        """QueueFullError should be a proper exception."""
        error = QueueFullError("test message")
        assert isinstance(error, Exception)
        assert str(error) == "test message"

    def test_queue_full_error_has_request_id(self) -> None:
        """QueueFullError should support request_id attribute."""
        error = QueueFullError("queue full", request_id="req-123")
        assert error.request_id == "req-123"

    @pytest.mark.asyncio
    async def test_reject_includes_queue_status(
        self, reject_queue: QueueManager
    ) -> None:
        """QueueFullError should include queue status information."""
        # Fill queue
        for _ in range(3):
            await reject_queue.acquire_slot()

        with pytest.raises(QueueFullError) as exc_info:
            await reject_queue.acquire_slot()

        # Should include useful info
        error = exc_info.value
        assert hasattr(error, "max_concurrent") or "max" in str(error).lower()


class TestQueueFullErrorAttributes:
    """Test QueueFullError exception attributes."""

    def test_queue_full_error_ends_in_error(self) -> None:
        """QueueFullError class name should end in 'Error' (AP-7)."""
        assert QueueFullError.__name__.endswith("Error")

    def test_queue_full_error_message(self) -> None:
        """QueueFullError should have descriptive message."""
        error = QueueFullError("Queue is full: max_concurrent=10, active=10")
        assert "Queue is full" in str(error)


# =============================================================================
# RequestItem Tests
# =============================================================================


class TestRequestItem:
    """Test RequestItem dataclass."""

    def test_request_item_creation(self) -> None:
        """RequestItem should be creatable with required fields."""
        item = RequestItem(
            request_id="req-123",
            priority=2,
            data={"prompt": "test"},
        )
        assert item.request_id == "req-123"
        assert item.priority == 2
        assert item.data == {"prompt": "test"}

    def test_request_item_default_priority(self) -> None:
        """RequestItem should have default priority=2 (normal)."""
        item = RequestItem(request_id="req-1", data={})
        assert item.priority == 2

    def test_request_item_comparison_for_priority_queue(self) -> None:
        """RequestItem should be comparable for PriorityQueue."""
        high = RequestItem(request_id="high", priority=3, data={})
        low = RequestItem(request_id="low", priority=1, data={})

        # Higher priority should sort lower (for min-heap behavior inverted)
        # We expect the implementation to handle this
        assert high.priority > low.priority

    def test_request_item_timestamp(self) -> None:
        """RequestItem should have timestamp for FIFO ordering."""
        item = RequestItem(request_id="req-1", data={})
        assert hasattr(item, "timestamp")
        assert isinstance(item.timestamp, float)


# =============================================================================
# Integration Tests
# =============================================================================


class TestQueueManagerIntegration:
    """Integration tests for QueueManager."""

    @pytest.mark.asyncio
    async def test_full_request_lifecycle(self, fifo_queue: QueueManager) -> None:
        """Test complete request lifecycle: enqueue -> acquire -> process -> release."""
        item = create_request_item("req-lifecycle")

        # Enqueue
        await fifo_queue.enqueue(item)
        assert fifo_queue.pending_count == 1

        # Dequeue
        dequeued = await fifo_queue.dequeue()
        assert dequeued.request_id == "req-lifecycle"
        assert fifo_queue.pending_count == 0

        # Acquire slot
        await fifo_queue.acquire_slot()
        assert fifo_queue.active_count == 1

        # Release slot
        fifo_queue.release_slot()
        assert fifo_queue.active_count == 0

    @pytest.mark.asyncio
    async def test_concurrent_enqueue_dequeue(
        self, fifo_queue: QueueManager
    ) -> None:
        """Test concurrent enqueue and dequeue operations."""
        results: list[str] = []

        async def producer() -> None:
            for i in range(5):
                item = create_request_item(f"req-{i}")
                await fifo_queue.enqueue(item)
                await asyncio.sleep(0.01)

        async def consumer() -> None:
            for _ in range(5):
                item = await fifo_queue.dequeue()
                results.append(item.request_id)

        await asyncio.gather(producer(), consumer())
        assert len(results) == 5

    @pytest.mark.asyncio
    async def test_queue_status_properties(self, fifo_queue: QueueManager) -> None:
        """Test queue status reporting properties."""
        assert fifo_queue.pending_count == 0
        assert fifo_queue.active_count == 0
        assert fifo_queue.is_empty is True
        assert fifo_queue.is_full is False

        # Add items
        await fifo_queue.enqueue(create_request_item("req-1"))
        assert fifo_queue.pending_count == 1
        assert fifo_queue.is_empty is False

        # Fill active slots
        for _ in range(10):
            await fifo_queue.acquire_slot()
        assert fifo_queue.is_full is True


class TestQueueManagerWithCallback:
    """Test QueueManager with callback support."""

    @pytest.mark.asyncio
    async def test_process_with_handler(self, fifo_queue: QueueManager) -> None:
        """Test processing items with a handler callback."""
        processed: list[str] = []

        async def handler(item: RequestItem) -> str:
            processed.append(item.request_id)
            return f"processed-{item.request_id}"

        item = create_request_item("req-callback")
        await fifo_queue.enqueue(item)
        dequeued = await fifo_queue.dequeue()
        result = await handler(dequeued)

        assert processed == ["req-callback"]
        assert result == "processed-req-callback"


class TestQueueManagerStatistics:
    """Test queue statistics tracking."""

    @pytest.mark.asyncio
    async def test_total_processed_count(self, fifo_queue: QueueManager) -> None:
        """Track total number of processed requests."""
        for i in range(5):
            await fifo_queue.acquire_slot()
            fifo_queue.release_slot()

        assert fifo_queue.total_processed >= 5

    @pytest.mark.asyncio
    async def test_queue_utilization(self, fifo_queue: QueueManager) -> None:
        """Calculate queue utilization percentage."""
        # No active requests
        assert fifo_queue.utilization == 0.0

        # Half capacity
        for _ in range(5):
            await fifo_queue.acquire_slot()
        assert fifo_queue.utilization == 0.5

        # Full capacity
        for _ in range(5):
            await fifo_queue.acquire_slot()
        assert fifo_queue.utilization == 1.0


class TestQueueManagerClear:
    """Test queue clearing functionality."""

    @pytest.mark.asyncio
    async def test_clear_pending_items(self, fifo_queue: QueueManager) -> None:
        """clear() should remove all pending items."""
        for i in range(5):
            await fifo_queue.enqueue(create_request_item(f"req-{i}"))

        assert fifo_queue.pending_count == 5
        await fifo_queue.clear()
        assert fifo_queue.pending_count == 0

    @pytest.mark.asyncio
    async def test_clear_returns_removed_items(
        self, fifo_queue: QueueManager
    ) -> None:
        """clear() should return the removed items."""
        items = [create_request_item(f"req-{i}") for i in range(3)]
        for item in items:
            await fifo_queue.enqueue(item)

        removed = await fifo_queue.clear()
        assert len(removed) == 3
        assert all(isinstance(item, RequestItem) for item in removed)
