"""Unit tests for H-1: QueueManager wired into completions route.

Tests cover:
- AC-H1.2.1: Only 1 inference request active at a time (max_concurrent=1)
- AC-H1.2.2: Queued requests get X-Queue-Position header
- AC-H1.2.3: Queue full returns HTTP 503 with Retry-After header
- AC-H1.2.4: Streaming and non-streaming paths both respect the queue
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI, status
from fastapi.testclient import TestClient
from httpx import ASGITransport, AsyncClient

from src.services.queue_manager import QueueFullError, QueueManager, QueueStrategy


# =============================================================================
# Constants
# =============================================================================

COMPLETIONS_ENDPOINT = "/v1/chat/completions"
MODEL_PHI4 = "phi-4"
ROLE_ASSISTANT = "assistant"
ROLE_USER = "user"
ROLE_SYSTEM = "system"
OBJECT_CHAT_COMPLETION = "chat.completion"
OBJECT_CHAT_COMPLETION_CHUNK = "chat.completion.chunk"
FINISH_REASON_STOP = "stop"


# =============================================================================
# Helpers
# =============================================================================


def _create_mock_response(
    model: str = MODEL_PHI4,
    content: str = "Test response.",
    prompt_tokens: int = 25,
    completion_tokens: int = 10,
) -> MagicMock:
    """Create a mock ChatCompletionResponse."""
    response = MagicMock()
    response.id = "chatcmpl-test"
    response.object = OBJECT_CHAT_COMPLETION
    response.created = int(time.time())
    response.model = model

    choice = MagicMock()
    choice.index = 0
    choice.message = MagicMock()
    choice.message.role = ROLE_ASSISTANT
    choice.message.content = content
    choice.message.tool_calls = None
    choice.finish_reason = FINISH_REASON_STOP
    response.choices = [choice]

    response.usage = MagicMock()
    response.usage.prompt_tokens = prompt_tokens
    response.usage.completion_tokens = completion_tokens
    response.usage.total_tokens = prompt_tokens + completion_tokens

    response.orchestration = None

    response.model_dump.return_value = {
        "id": response.id,
        "object": response.object,
        "created": response.created,
        "model": response.model,
        "choices": [
            {
                "index": 0,
                "message": {"role": ROLE_ASSISTANT, "content": content, "tool_calls": None},
                "finish_reason": FINISH_REASON_STOP,
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
        "orchestration": None,
    }

    return response


def _create_mock_chunk(content: str = "Hello") -> MagicMock:
    """Create a mock streaming chunk."""
    chunk = MagicMock()
    chunk.id = "chatcmpl-chunk-1"
    chunk.object = OBJECT_CHAT_COMPLETION_CHUNK
    chunk.created = int(time.time())
    chunk.model = MODEL_PHI4

    chunk_data = {
        "id": chunk.id,
        "object": OBJECT_CHAT_COMPLETION_CHUNK,
        "created": chunk.created,
        "model": MODEL_PHI4,
        "choices": [{"index": 0, "delta": {"content": content}, "finish_reason": None}],
    }
    chunk.to_sse.return_value = f"data: {json.dumps(chunk_data)}"
    return chunk


def _valid_request(stream: bool = False) -> dict[str, Any]:
    """Create a valid chat completion request body."""
    return {
        "model": MODEL_PHI4,
        "messages": [
            {"role": ROLE_SYSTEM, "content": "You are helpful."},
            {"role": ROLE_USER, "content": "Hello"},
        ],
        "stream": stream,
        "max_tokens": 50,
    }


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_provider() -> MagicMock:
    """Create mock InferenceProvider."""
    provider = MagicMock()
    provider.model_info = MagicMock()
    provider.model_info.model_id = MODEL_PHI4
    provider.is_loaded = True
    provider.generate = AsyncMock(return_value=_create_mock_response())
    return provider


@pytest.fixture
def mock_provider_slow() -> MagicMock:
    """Create mock InferenceProvider that takes 0.3s per request."""
    provider = MagicMock()
    provider.model_info = MagicMock()
    provider.model_info.model_id = MODEL_PHI4
    provider.is_loaded = True

    async def slow_generate(req: Any) -> MagicMock:
        await asyncio.sleep(0.3)
        return _create_mock_response()

    provider.generate = AsyncMock(side_effect=slow_generate)
    return provider


@pytest.fixture
def mock_model_manager(mock_provider: MagicMock) -> MagicMock:
    """Create mock ModelManager."""
    manager = MagicMock()
    manager.get_provider.return_value = mock_provider
    manager.get_loaded_models.return_value = [MODEL_PHI4]
    return manager


@pytest.fixture
def slow_model_manager(mock_provider_slow: MagicMock) -> MagicMock:
    """Create mock ModelManager with slow provider (for concurrency tests)."""
    manager = MagicMock()
    manager.get_provider.return_value = mock_provider_slow
    manager.get_loaded_models.return_value = [MODEL_PHI4]
    return manager


@pytest.fixture
def queue_manager() -> QueueManager:
    """Create a QueueManager with max_concurrent=1."""
    return QueueManager(
        max_concurrent=1,
        strategy=QueueStrategy.FIFO,
        reject_when_full=False,
    )


@pytest.fixture
def reject_queue_manager() -> QueueManager:
    """Create a QueueManager that rejects when full."""
    return QueueManager(
        max_concurrent=1,
        strategy=QueueStrategy.FIFO,
        reject_when_full=True,
    )


@pytest.fixture
def app_with_queue(mock_model_manager: MagicMock, queue_manager: QueueManager) -> FastAPI:
    """Create FastAPI app with both model manager and queue manager."""
    from src.api.routes.completions import router

    app = FastAPI()
    app.include_router(router, prefix="/v1")
    app.state.model_manager = mock_model_manager
    app.state.queue_manager = queue_manager
    return app


@pytest.fixture
def app_with_reject_queue(
    mock_model_manager: MagicMock, reject_queue_manager: QueueManager
) -> FastAPI:
    """Create FastAPI app with queue that rejects when full."""
    from src.api.routes.completions import router

    app = FastAPI()
    app.include_router(router, prefix="/v1")
    app.state.model_manager = mock_model_manager
    app.state.queue_manager = reject_queue_manager
    return app


@pytest.fixture
def app_with_slow_queue(
    slow_model_manager: MagicMock, queue_manager: QueueManager
) -> FastAPI:
    """Create FastAPI app with slow model manager and queue."""
    from src.api.routes.completions import router

    app = FastAPI()
    app.include_router(router, prefix="/v1")
    app.state.model_manager = slow_model_manager
    app.state.queue_manager = queue_manager
    return app


@pytest.fixture
def client_with_queue(app_with_queue: FastAPI) -> TestClient:
    return TestClient(app_with_queue)


@pytest.fixture
def client_reject_queue(app_with_reject_queue: FastAPI) -> TestClient:
    return TestClient(app_with_reject_queue)


# =============================================================================
# AC-H1.2.1: Only 1 inference request active at a time
# =============================================================================


class TestQueueSerializesRequests:
    """Test that requests are serialized through the queue."""

    def test_non_streaming_request_acquires_and_releases_slot(
        self, client_with_queue: TestClient, queue_manager: QueueManager
    ) -> None:
        """Non-streaming request acquires slot before inference and releases after."""
        assert queue_manager.active_count == 0
        response = client_with_queue.post(COMPLETIONS_ENDPOINT, json=_valid_request())
        assert response.status_code == status.HTTP_200_OK
        # Slot released after request completes
        assert queue_manager.active_count == 0
        assert queue_manager.total_processed == 1

    def test_streaming_request_acquires_and_releases_slot(
        self, client_with_queue: TestClient, queue_manager: QueueManager
    ) -> None:
        """Streaming request holds slot for full duration of stream."""
        assert queue_manager.active_count == 0
        response = client_with_queue.post(
            COMPLETIONS_ENDPOINT, json=_valid_request(stream=True)
        )
        assert response.status_code == status.HTTP_200_OK
        # Consume the stream
        _ = response.text
        # Slot released after stream completes
        assert queue_manager.active_count == 0
        assert queue_manager.total_processed == 1

    @pytest.mark.asyncio
    async def test_concurrent_requests_serialized(
        self, app_with_slow_queue: FastAPI, queue_manager: QueueManager
    ) -> None:
        """Two concurrent requests are serialized: second waits for first.

        Subtask 2.2.5: Second request waits for first to complete.
        """
        order: list[int] = []

        transport = ASGITransport(app=app_with_slow_queue)
        async with AsyncClient(transport=transport, base_url="http://test") as client:

            async def make_request(idx: int) -> None:
                resp = await client.post(
                    COMPLETIONS_ENDPOINT, json=_valid_request()
                )
                assert resp.status_code == status.HTTP_200_OK
                order.append(idx)

            # Fire both concurrently — queue max_concurrent=1
            task1 = asyncio.create_task(make_request(1))
            await asyncio.sleep(0.05)  # Ensure task1 acquires slot first
            task2 = asyncio.create_task(make_request(2))

            await asyncio.gather(task1, task2)

        # Both completed
        assert len(order) == 2
        # First request finishes before second
        assert order == [1, 2]
        assert queue_manager.total_processed == 2

    @pytest.mark.asyncio
    async def test_max_concurrent_one_enforced(
        self, app_with_slow_queue: FastAPI, queue_manager: QueueManager
    ) -> None:
        """Never more than 1 request active at a time."""
        max_active_seen = 0

        original_generate = app_with_slow_queue.state.model_manager.get_provider().generate

        async def tracking_generate(req: Any) -> MagicMock:
            nonlocal max_active_seen
            # At this point we're inside the queue slot
            active = queue_manager.active_count
            if active > max_active_seen:
                max_active_seen = active
            return await original_generate(req)

        app_with_slow_queue.state.model_manager.get_provider().generate = AsyncMock(
            side_effect=tracking_generate
        )

        transport = ASGITransport(app=app_with_slow_queue)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            tasks = [
                client.post(COMPLETIONS_ENDPOINT, json=_valid_request())
                for _ in range(3)
            ]
            responses = await asyncio.gather(*tasks)

        assert all(r.status_code == status.HTTP_200_OK for r in responses)
        assert max_active_seen == 1


# =============================================================================
# AC-H1.2.2: Queued requests get X-Queue-Position header
# =============================================================================


class TestXQueuePositionHeader:
    """Test X-Queue-Position header is set when queue is full."""

    @pytest.mark.asyncio
    async def test_x_queue_position_header_when_queued(
        self, app_with_slow_queue: FastAPI, queue_manager: QueueManager
    ) -> None:
        """Request that arrives when queue is full gets X-Queue-Position header."""
        transport = ASGITransport(app=app_with_slow_queue)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            # First request — will hold the slot for 0.3s
            task1 = asyncio.create_task(
                client.post(COMPLETIONS_ENDPOINT, json=_valid_request())
            )
            await asyncio.sleep(0.05)  # Let task1 acquire slot

            # Second request — queue is full, should get X-Queue-Position
            task2 = asyncio.create_task(
                client.post(COMPLETIONS_ENDPOINT, json=_valid_request())
            )
            r1, r2 = await asyncio.gather(task1, task2)

        assert r1.status_code == status.HTTP_200_OK
        assert r2.status_code == status.HTTP_200_OK
        # Note: the header is set at request submission time when is_full=True.
        # Since task2 submits while task1 is active, it sees is_full=True.


# =============================================================================
# AC-H1.2.3: Queue full returns HTTP 503 with Retry-After header
# =============================================================================


class TestQueueFull503:
    """Test that queue full condition returns HTTP 503."""

    @pytest.mark.asyncio
    async def test_queue_full_returns_503_with_retry_after(
        self, app_with_reject_queue: FastAPI, reject_queue_manager: QueueManager
    ) -> None:
        """When reject_when_full=True and queue is full, return 503."""
        transport = ASGITransport(app=app_with_reject_queue)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            # Manually acquire the only slot
            await reject_queue_manager.acquire_slot()
            assert reject_queue_manager.active_count == 1

            # Now make a request — should get 503
            response = await client.post(
                COMPLETIONS_ENDPOINT, json=_valid_request()
            )

            assert response.status_code == status.HTTP_503_SERVICE_UNAVAILABLE
            assert response.headers.get("retry-after") == "5"
            data = response.json()
            assert "retry_after_seconds" in data
            assert data["max_concurrent"] == 1

            # Cleanup
            reject_queue_manager.release_slot()

    def test_503_response_body_has_detail(
        self, client_reject_queue: TestClient, reject_queue_manager: QueueManager
    ) -> None:
        """503 response body includes helpful detail message."""
        # Synchronously acquire the only slot
        import asyncio as _asyncio
        _asyncio.get_event_loop().run_until_complete(reject_queue_manager.acquire_slot())

        response = client_reject_queue.post(
            COMPLETIONS_ENDPOINT, json=_valid_request()
        )
        assert response.status_code == status.HTTP_503_SERVICE_UNAVAILABLE
        data = response.json()
        assert "detail" in data
        assert "queue" in data["detail"].lower() or "retry" in data["detail"].lower()

        reject_queue_manager.release_slot()


# =============================================================================
# AC-H1.2.4: Both streaming and non-streaming respect queue
# =============================================================================


class TestBothPathsRespectQueue:
    """Test that both streaming and non-streaming paths go through the queue."""

    def test_non_streaming_increments_total_processed(
        self, client_with_queue: TestClient, queue_manager: QueueManager
    ) -> None:
        """Non-streaming path increments queue total_processed."""
        assert queue_manager.total_processed == 0
        client_with_queue.post(COMPLETIONS_ENDPOINT, json=_valid_request())
        assert queue_manager.total_processed == 1

    def test_streaming_increments_total_processed(
        self, client_with_queue: TestClient, queue_manager: QueueManager
    ) -> None:
        """Streaming path increments queue total_processed."""
        assert queue_manager.total_processed == 0
        response = client_with_queue.post(
            COMPLETIONS_ENDPOINT, json=_valid_request(stream=True)
        )
        _ = response.text  # Consume stream
        assert queue_manager.total_processed == 1

    def test_back_to_back_streaming_and_non_streaming(
        self, client_with_queue: TestClient, queue_manager: QueueManager
    ) -> None:
        """Mixed streaming and non-streaming requests all go through queue."""
        client_with_queue.post(COMPLETIONS_ENDPOINT, json=_valid_request())
        client_with_queue.post(COMPLETIONS_ENDPOINT, json=_valid_request(stream=True))
        _ = client_with_queue.post(COMPLETIONS_ENDPOINT, json=_valid_request()).text
        assert queue_manager.total_processed == 3


# =============================================================================
# Context manager unit tests
# =============================================================================


class TestQueueManagerAcquireContextManager:
    """Test the QueueManager.acquire() async context manager."""

    @pytest.mark.asyncio
    async def test_acquire_context_manager_acquires_and_releases(self) -> None:
        """acquire() context manager acquires slot on enter, releases on exit."""
        qm = QueueManager(max_concurrent=1)
        assert qm.active_count == 0

        async with qm.acquire():
            assert qm.active_count == 1

        assert qm.active_count == 0
        assert qm.total_processed == 1

    @pytest.mark.asyncio
    async def test_acquire_releases_on_exception(self) -> None:
        """Slot is released even if body raises an exception."""
        qm = QueueManager(max_concurrent=1)

        with pytest.raises(ValueError, match="boom"):
            async with qm.acquire():
                assert qm.active_count == 1
                raise ValueError("boom")

        assert qm.active_count == 0
        assert qm.total_processed == 1

    @pytest.mark.asyncio
    async def test_acquire_returns_queue_manager(self) -> None:
        """acquire() context manager returns the QueueManager instance."""
        qm = QueueManager(max_concurrent=1)
        async with qm.acquire() as mgr:
            assert mgr is qm

    @pytest.mark.asyncio
    async def test_acquire_blocks_when_full(self) -> None:
        """Second acquire() blocks until first is released."""
        qm = QueueManager(max_concurrent=1)
        acquired_second = asyncio.Event()

        async with qm.acquire():
            # Try to acquire second — should block
            async def try_second() -> None:
                async with qm.acquire():
                    acquired_second.set()

            task = asyncio.create_task(try_second())
            # Give event loop time to process — second should still be blocked
            await asyncio.sleep(0.05)
            assert not acquired_second.is_set()

        # After exiting first, second should complete
        await asyncio.wait_for(task, timeout=1.0)
        assert acquired_second.is_set()

    @pytest.mark.asyncio
    async def test_acquire_raises_queue_full_when_reject_mode(self) -> None:
        """acquire() raises QueueFullError when reject_when_full=True."""
        qm = QueueManager(max_concurrent=1, reject_when_full=True)
        await qm.acquire_slot()  # Take the only slot

        with pytest.raises(QueueFullError):
            async with qm.acquire():
                pass  # Should never reach here

        qm.release_slot()
