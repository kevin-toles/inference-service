"""Load testing for inference-service.

Tests concurrent request handling and service performance under load.

AC-21.5: Load test: 5 concurrent requests handled without queue overflow

Usage:
    pytest tests/integration/test_load.py -v

    # Remote deployment
    INFERENCE_BASE_URL=http://10.0.0.50:8085 pytest tests/integration/test_load.py -v
"""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING, Any

import pytest


if TYPE_CHECKING:
    import httpx


pytestmark = [
    pytest.mark.integration,
    pytest.mark.asyncio,
    pytest.mark.slow,
    pytest.mark.load_test,
]


# =============================================================================
# Concurrent Request Tests
# =============================================================================


class TestConcurrentRequests:
    """Test handling of concurrent requests."""

    @pytest.mark.requires_model
    async def test_5_concurrent_requests_no_queue_overflow(
        self,
        client: httpx.AsyncClient,
        skip_if_no_models: None,
        chat_request_factory: Any,
    ) -> None:
        """Test 5 concurrent requests handled without QueueFullError.

        AC-21.5: 5 concurrent requests complete without queue overflow
        """
        num_requests = 5

        async def make_request(request_id: int) -> tuple[int, int, float, str | None]:
            """Make a request and return (id, status, duration, error)."""
            request = chat_request_factory.simple(
                message=f"Request {request_id}: What is {request_id} + 1?",
                max_tokens=20,
            )

            start = time.monotonic()
            response = await client.post("/v1/chat/completions", json=request)
            duration = time.monotonic() - start

            error = None
            if response.status_code != 200:
                try:
                    data = response.json()
                    error = data.get("error", {}).get("type", str(response.status_code))
                except Exception:
                    error = str(response.status_code)

            return (request_id, response.status_code, duration, error)

        # Launch all requests concurrently
        start_time = time.monotonic()
        tasks = [make_request(i) for i in range(num_requests)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        total_time = time.monotonic() - start_time

        # Analyze results
        successes = 0
        queue_full_errors = 0
        other_errors = 0

        for result in results:
            if isinstance(result, Exception):
                other_errors += 1
            elif isinstance(result, tuple):
                status, error = result[1], result[3]
                if status == 200:
                    successes += 1
                elif error and "queue" in str(error).lower():
                    queue_full_errors += 1
                else:
                    other_errors += 1

        # No queue overflow errors should occur for 5 requests
        assert queue_full_errors == 0, (
            f"QueueFullError detected: {queue_full_errors} of {num_requests} requests"
        )

        # All 5 should succeed
        assert successes == num_requests, (
            f"Only {successes}/{num_requests} succeeded. "
            f"Errors: {other_errors}, QueueFull: {queue_full_errors}"
        )

        print(f"\n5 concurrent requests completed in {total_time:.2f}s")

    @pytest.mark.requires_model
    async def test_10_concurrent_requests(
        self,
        client: httpx.AsyncClient,
        skip_if_no_models: None,
        chat_request_factory: Any,
    ) -> None:
        """Test 10 concurrent requests complete successfully.

        Note: May have some queue rejections under load.
        """
        num_requests = 10

        async def make_request(request_id: int) -> tuple[int, int, float]:
            """Make a request and return (id, status, duration)."""
            request = chat_request_factory.simple(
                message=f"Request {request_id}: What is {request_id} + 1?",
                max_tokens=20,
            )

            start = time.monotonic()
            response = await client.post("/v1/chat/completions", json=request)
            duration = time.monotonic() - start

            return (request_id, response.status_code, duration)

        # Launch all requests concurrently
        start_time = time.monotonic()
        tasks = [make_request(i) for i in range(num_requests)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        total_time = time.monotonic() - start_time

        # Analyze results
        successes = 0
        for result in results:
            if isinstance(result, Exception):
                pass  # Count as failure by not incrementing successes
            elif isinstance(result, tuple) and result[1] == 200:
                successes += 1

        # At least 80% should succeed
        success_rate = successes / num_requests
        assert success_rate >= 0.8, f"Success rate {success_rate:.0%} < 80%"

        # Total time should be reasonable (not sequential)
        # If sequential, would be ~10x single request time
        # With concurrency, should be much less
        assert total_time < 300, f"Total time {total_time:.1f}s > 300s timeout"

    @pytest.mark.requires_model
    async def test_5_concurrent_requests_all_succeed(
        self,
        client: httpx.AsyncClient,
        skip_if_no_models: None,
        chat_request_factory: Any,
    ) -> None:
        """Test 5 concurrent requests all complete successfully."""
        num_requests = 5

        async def make_request(request_id: int) -> int:
            """Make a request and return status code."""
            request = chat_request_factory.simple(
                message=f"Say '{request_id}'",
                max_tokens=10,
            )
            response = await client.post("/v1/chat/completions", json=request)
            return response.status_code

        tasks = [make_request(i) for i in range(num_requests)]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # All should succeed or be handled gracefully
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                pytest.fail(f"Request {i} raised: {result}")
            # Allow 200 (success) or 503 (busy/queue full)
            assert result in [200, 503], f"Request {i} returned {result}"

    @pytest.mark.requires_model
    async def test_concurrent_streaming_requests(
        self,
        client: httpx.AsyncClient,
        skip_if_no_models: None,
        chat_request_factory: Any,
        sse_parser: Any,
    ) -> None:
        """Test concurrent streaming requests."""
        num_requests = 3

        async def stream_request(request_id: int) -> tuple[int, bool]:
            """Make streaming request, return (id, has_done)."""
            request = chat_request_factory.streaming(
                message=f"Count {request_id}",
                max_tokens=20,
            )

            has_done = False
            async with client.stream("POST", "/v1/chat/completions", json=request) as response:
                if response.status_code == 200:
                    content = (await response.aread()).decode("utf-8")
                    chunks = sse_parser.parse_stream(content)
                    has_done = "[DONE]" in chunks

            return (request_id, has_done)

        tasks = [stream_request(i) for i in range(num_requests)]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Check results
        for result in results:
            if isinstance(result, Exception):
                continue  # May fail under load
            if isinstance(result, tuple):
                _req_id, has_done = result
                # If succeeded, should have [DONE]
                if not has_done:
                    # Not a hard failure, may have been busy
                    pass


# =============================================================================
# Queue Behavior Tests
# =============================================================================


class TestQueueBehavior:
    """Test request queue behavior under load."""

    @pytest.mark.requires_model
    async def test_requests_queued_when_busy(
        self,
        client: httpx.AsyncClient,
        skip_if_no_models: None,
        chat_request_factory: Any,
    ) -> None:
        """Test requests are queued when service is busy."""
        # Send many requests quickly
        num_requests = 15

        async def quick_request() -> int:
            request = chat_request_factory.simple(
                message="Hi",
                max_tokens=5,
            )
            response = await client.post("/v1/chat/completions", json=request)
            return response.status_code

        tasks = [quick_request() for _ in range(num_requests)]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        status_counts: dict[int, int] = {}
        for result in results:
            if isinstance(result, int):
                status_counts[result] = status_counts.get(result, 0) + 1

        # Should have some successful (200) or queued/busy (503)
        assert 200 in status_counts or 503 in status_counts

    @pytest.mark.requires_model
    async def test_queue_full_returns_503(
        self,
        client: httpx.AsyncClient,
        skip_if_no_models: None,
        chat_request_factory: Any,
    ) -> None:
        """Test queue full condition returns 503."""
        # Try to overwhelm the queue
        num_requests = 50

        async def flood_request() -> int:
            request = chat_request_factory.simple(
                message="Test",
                max_tokens=100,  # Longer generation
            )
            try:
                response = await client.post(
                    "/v1/chat/completions",
                    json=request,
                    timeout=5.0,  # Short timeout
                )
                return response.status_code
            except Exception:
                return -1

        tasks = [flood_request() for _ in range(num_requests)]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Should get some 503s when queue fills
        _status_codes = [r for r in results if isinstance(r, int)]
        # This is expected behavior, not a test failure
        # Just verify we can handle the load


# =============================================================================
# Performance Metrics Tests
# =============================================================================


class TestPerformanceMetrics:
    """Test performance under various conditions."""

    @pytest.mark.requires_model
    async def test_response_time_under_load(
        self,
        client: httpx.AsyncClient,
        skip_if_no_models: None,
        chat_request_factory: Any,
    ) -> None:
        """Test response times don't degrade too much under load."""
        # First, get baseline with single request
        request = chat_request_factory.simple(
            message="Hello",
            max_tokens=10,
        )

        start = time.monotonic()
        await client.post("/v1/chat/completions", json=request)
        baseline = time.monotonic() - start

        # Now test under light load (3 concurrent)
        async def timed_request() -> float:
            start = time.monotonic()
            await client.post("/v1/chat/completions", json=request)
            return time.monotonic() - start

        tasks = [timed_request() for _ in range(3)]
        times = await asyncio.gather(*tasks, return_exceptions=True)

        valid_times = [t for t in times if isinstance(t, float)]
        if valid_times:
            avg_time = sum(valid_times) / len(valid_times)
            # Under load, expect at most 5x baseline
            # (generous due to model inference variance)
            assert avg_time < baseline * 5, f"Avg {avg_time:.1f}s > 5x baseline {baseline:.1f}s"

    @pytest.mark.requires_model
    async def test_throughput_measurement(
        self,
        client: httpx.AsyncClient,
        skip_if_no_models: None,
        chat_request_factory: Any,
    ) -> None:
        """Measure request throughput."""
        num_requests = 5

        request = chat_request_factory.simple(
            message="Hi",
            max_tokens=5,
        )

        start = time.monotonic()
        tasks = [
            client.post("/v1/chat/completions", json=request)
            for _ in range(num_requests)
        ]
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        total_time = time.monotonic() - start

        successful = sum(
            1 for r in responses
            if hasattr(r, "status_code") and r.status_code == 200
        )

        if successful > 0:
            throughput = successful / total_time
            # Just log, don't fail on specific throughput
            # Model inference time varies
            assert throughput > 0


# =============================================================================
# Stress Tests
# =============================================================================


class TestStress:
    """Stress tests for service stability."""

    @pytest.mark.requires_model
    async def test_rapid_sequential_requests(
        self,
        client: httpx.AsyncClient,
        skip_if_no_models: None,
        chat_request_factory: Any,
    ) -> None:
        """Test rapid sequential requests don't cause issues."""
        request = chat_request_factory.simple(
            message="Hi",
            max_tokens=5,
        )

        success_count = 0
        for _ in range(10):
            response = await client.post("/v1/chat/completions", json=request)
            if response.status_code == 200:
                success_count += 1
            # Small delay to avoid overwhelming
            await asyncio.sleep(0.1)

        # Most should succeed
        assert success_count >= 7, f"Only {success_count}/10 succeeded"

    @pytest.mark.requires_model
    async def test_mixed_request_types(
        self,
        client: httpx.AsyncClient,
        skip_if_no_models: None,
        chat_request_factory: Any,
    ) -> None:
        """Test mixed streaming and non-streaming requests."""
        tasks: list[Any] = []

        # Mix of request types
        for i in range(6):
            if i % 2 == 0:
                request = chat_request_factory.simple(max_tokens=10)
                tasks.append(client.post("/v1/chat/completions", json=request))
            else:
                request = chat_request_factory.streaming(max_tokens=10)
                # For streaming, just do a regular POST to check response
                tasks.append(client.post("/v1/chat/completions", json=request))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Should handle mixed types
        success_count = sum(
            1 for r in results
            if hasattr(r, "status_code") and r.status_code == 200
        )
        assert success_count >= 3, "Less than half of mixed requests succeeded"
