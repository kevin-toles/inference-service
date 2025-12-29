"""Queue Manager service for request queuing and concurrency control.

This module implements:
- AC-17.1: QueueManager uses asyncio.Queue
- AC-17.2: QueueManager enforces max_concurrent_requests
- AC-17.3: QueueManager supports FIFO and priority strategies
- AC-17.4: QueueManager rejects when full if configured

Reference: ARCHITECTURE.md → Concurrency & Scaling
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class QueueStrategy(str, Enum):
    """Queue processing strategy enumeration.

    Values:
        FIFO: First-in-first-out processing order
        PRIORITY: Higher priority (3) processed before lower (1)
    """

    FIFO = "fifo"
    PRIORITY = "priority"


class QueueFullError(Exception):
    """Exception raised when queue is full and reject_when_full=True.

    Attributes:
        request_id: The ID of the request that was rejected
        max_concurrent: Maximum concurrent requests allowed
        message: Error message
    """

    def __init__(
        self,
        message: str,
        request_id: str | None = None,
        max_concurrent: int | None = None,
    ) -> None:
        """Initialize QueueFullError.

        Args:
            message: Error description
            request_id: Optional ID of rejected request
            max_concurrent: Optional max concurrent limit
        """
        super().__init__(message)
        self.request_id = request_id
        self.max_concurrent = max_concurrent


@dataclass
class RequestItem:
    """Request item for queue storage.

    Attributes:
        request_id: Unique identifier for the request
        priority: Priority level (1=low, 2=normal, 3=high)
        data: Request data payload
        timestamp: Creation time for FIFO ordering
    """

    request_id: str
    data: dict[str, Any]
    priority: int = 2
    timestamp: float = field(default_factory=time.time)

    def __lt__(self, other: RequestItem) -> bool:
        """Compare for priority queue ordering.

        Higher priority values should be processed first, so we invert
        the comparison. For same priority, earlier timestamp wins.
        """
        if self.priority != other.priority:
            # Higher priority (3) should come before lower (1)
            # PriorityQueue is a min-heap, so invert comparison
            return self.priority > other.priority
        # Same priority: FIFO by timestamp
        return self.timestamp < other.timestamp


class QueueManager:
    """Manages request queuing with concurrency control.

    Implements asyncio-based request queuing with support for FIFO and
    priority-based processing strategies. Enforces max concurrent requests
    and optionally rejects requests when full.

    Attributes:
        max_concurrent: Maximum number of concurrent requests
        strategy: Queue processing strategy (FIFO or PRIORITY)
        reject_when_full: Whether to reject requests when at capacity
    """

    def __init__(
        self,
        max_concurrent: int = 10,
        strategy: QueueStrategy = QueueStrategy.FIFO,
        reject_when_full: bool = False,
    ) -> None:
        """Initialize QueueManager.

        Args:
            max_concurrent: Maximum concurrent requests (default: 10)
            strategy: Queue strategy (default: FIFO)
            reject_when_full: Reject requests when full (default: False)
        """
        self._max_concurrent = max_concurrent
        self._strategy = strategy
        self._reject_when_full = reject_when_full

        # Create appropriate queue type based on strategy
        if strategy == QueueStrategy.PRIORITY:
            self._queue: asyncio.Queue[RequestItem] = asyncio.PriorityQueue()
        else:
            self._queue = asyncio.Queue()

        # Semaphore for concurrency control
        self._semaphore = asyncio.Semaphore(max_concurrent)

        # Track active requests
        self._active_count = 0
        self._active_lock = asyncio.Lock()

        # Statistics
        self._total_processed = 0

    @property
    def max_concurrent(self) -> int:
        """Get maximum concurrent requests limit."""
        return self._max_concurrent

    @property
    def active_count(self) -> int:
        """Get current number of active requests."""
        return self._active_count

    @property
    def pending_count(self) -> int:
        """Get number of pending requests in queue."""
        return self._queue.qsize()

    @property
    def is_empty(self) -> bool:
        """Check if queue has no pending items."""
        return self._queue.empty()

    @property
    def is_full(self) -> bool:
        """Check if at maximum concurrent capacity."""
        return self._active_count >= self._max_concurrent

    @property
    def total_processed(self) -> int:
        """Get total number of requests processed."""
        return self._total_processed

    @property
    def utilization(self) -> float:
        """Calculate queue utilization as percentage (0.0 to 1.0)."""
        if self._max_concurrent == 0:
            return 0.0
        return self._active_count / self._max_concurrent

    async def enqueue(self, item: RequestItem) -> None:
        """Add a request item to the queue.

        Args:
            item: The request item to enqueue
        """
        await self._queue.put(item)

    async def dequeue(self) -> RequestItem:
        """Remove and return the next request item from the queue.

        For FIFO strategy, returns items in insertion order.
        For PRIORITY strategy, returns highest priority items first.

        Returns:
            The next RequestItem to process
        """
        return await self._queue.get()

    async def acquire_slot(self) -> None:
        """Acquire a processing slot.

        If reject_when_full is True and queue is full, raises QueueFullError.
        Otherwise, blocks until a slot becomes available.

        Raises:
            QueueFullError: If reject_when_full=True and at capacity
        """
        if self._reject_when_full:
            # Try to acquire without blocking
            async with self._active_lock:
                if self._active_count >= self._max_concurrent:
                    raise QueueFullError(
                        f"Queue is full: max_concurrent={self._max_concurrent}, "
                        f"active={self._active_count}",
                        max_concurrent=self._max_concurrent,
                    )
                self._active_count += 1
                return

        # Block until slot available
        await self._semaphore.acquire()
        async with self._active_lock:
            self._active_count += 1

    def release_slot(self) -> None:
        """Release a processing slot.

        Should be called after request processing completes.
        """
        # Decrement active count
        # Note: Using sync context manager pattern for simplicity
        # In production, consider async lock if needed
        self._active_count = max(0, self._active_count - 1)
        self._total_processed += 1

        # Release semaphore if not in reject mode
        if not self._reject_when_full:
            self._semaphore.release()

    def clear(self) -> list[RequestItem]:
        """Clear all pending items from the queue.

        Returns:
            List of removed RequestItem instances
        """
        removed: list[RequestItem] = []

        while not self._queue.empty():
            try:
                item = self._queue.get_nowait()
                removed.append(item)
            except asyncio.QueueEmpty:
                break

        return removed
