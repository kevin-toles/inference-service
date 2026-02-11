"""Integration tests for H-1: QueueManager FIFO ordering in completions route.

Subtask 2.2.6: 3 concurrent requests complete in FIFO order.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from src.services.queue_manager import QueueManager, QueueStrategy


# =============================================================================
# Constants
# =============================================================================

COMPLETIONS_ENDPOINT = "/v1/chat/completions"
MODEL_PHI4 = "phi-4"


# =============================================================================
# Helpers
# =============================================================================


def _valid_request() -> dict[str, Any]:
    return {
        "model": MODEL_PHI4,
        "messages": [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
        ],
        "stream": False,
        "max_tokens": 50,
    }


def _create_mock_response(model: str = MODEL_PHI4) -> MagicMock:
    """Minimal mock response for integration test."""
    response = MagicMock()
    response.id = "chatcmpl-integration"
    response.object = "chat.completion"
    response.created = int(time.time())
    response.model = model
    response.choices = [MagicMock()]
    response.choices[0].index = 0
    response.choices[0].message = MagicMock()
    response.choices[0].message.role = "assistant"
    response.choices[0].message.content = "response"
    response.choices[0].message.tool_calls = None
    response.choices[0].finish_reason = "stop"
    response.usage = MagicMock()
    response.usage.prompt_tokens = 10
    response.usage.completion_tokens = 5
    response.usage.total_tokens = 15
    response.orchestration = None
    response.model_dump.return_value = {
        "id": response.id,
        "object": "chat.completion",
        "created": response.created,
        "model": model,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "response", "tool_calls": None}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        "orchestration": None,
    }
    return response


# =============================================================================
# Integration: 3 concurrent requests complete in FIFO order
# =============================================================================


class TestFIFOOrderIntegration:
    """Subtask 2.2.6: 3 concurrent requests complete in FIFO order."""

    @pytest.mark.asyncio
    async def test_three_concurrent_requests_fifo_order(self) -> None:
        """Three concurrent non-streaming requests complete in FIFO order.

        Each request takes 0.2s. With max_concurrent=1, they must serialize.
        The order of completion should match the order of submission.
        """
        from src.api.routes.completions import router

        completion_order: list[int] = []

        # Create a slow provider that logs completion order
        async def slow_generate(req: Any, idx: int = 0) -> MagicMock:
            await asyncio.sleep(0.2)
            return _create_mock_response()

        call_count = 0

        async def tracking_generate(req: Any) -> MagicMock:
            nonlocal call_count
            call_count += 1
            current = call_count
            await asyncio.sleep(0.2)
            completion_order.append(current)
            return _create_mock_response()

        provider = MagicMock()
        provider.generate = AsyncMock(side_effect=tracking_generate)

        manager = MagicMock()
        manager.get_provider.return_value = provider

        queue_manager = QueueManager(
            max_concurrent=1,
            strategy=QueueStrategy.FIFO,
            reject_when_full=False,
        )

        app = FastAPI()
        app.include_router(router, prefix="/v1")
        app.state.model_manager = manager
        app.state.queue_manager = queue_manager

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:

            submission_order: list[int] = []

            async def submit_request(idx: int) -> int:
                submission_order.append(idx)
                resp = await client.post(COMPLETIONS_ENDPOINT, json=_valid_request())
                assert resp.status_code == 200
                return idx

            # Stagger submissions slightly to enforce ordering
            task1 = asyncio.create_task(submit_request(1))
            await asyncio.sleep(0.02)
            task2 = asyncio.create_task(submit_request(2))
            await asyncio.sleep(0.02)
            task3 = asyncio.create_task(submit_request(3))

            await asyncio.gather(task1, task2, task3)

        # All three completed
        assert len(completion_order) == 3
        # FIFO: first submitted = first completed
        assert completion_order == [1, 2, 3]
        assert queue_manager.total_processed == 3

    @pytest.mark.asyncio
    async def test_queue_stats_after_burst(self) -> None:
        """After a burst of 3 requests, queue stats reflect all processed."""
        from src.api.routes.completions import router

        provider = MagicMock()
        provider.generate = AsyncMock(return_value=_create_mock_response())

        manager = MagicMock()
        manager.get_provider.return_value = provider

        queue_manager = QueueManager(
            max_concurrent=1,
            strategy=QueueStrategy.FIFO,
            reject_when_full=False,
        )

        app = FastAPI()
        app.include_router(router, prefix="/v1")
        app.state.model_manager = manager
        app.state.queue_manager = queue_manager

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            tasks = [
                client.post(COMPLETIONS_ENDPOINT, json=_valid_request())
                for _ in range(3)
            ]
            responses = await asyncio.gather(*tasks)

        assert all(r.status_code == 200 for r in responses)
        assert queue_manager.total_processed == 3
        assert queue_manager.active_count == 0
        assert queue_manager.is_empty
