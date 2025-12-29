"""Streaming integration tests for inference-service.

Tests SSE streaming responses and proper chunk formatting.

AC-21.2: Streaming test: SSE chunks received correctly
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import httpx
import pytest


if TYPE_CHECKING:
    pass


pytestmark = [pytest.mark.integration, pytest.mark.asyncio]


# =============================================================================
# SSE Streaming Tests
# =============================================================================


class TestSSEStreaming:
    """Test Server-Sent Events streaming responses."""

    @pytest.mark.requires_model
    async def test_streaming_returns_sse_format(
        self,
        client: httpx.AsyncClient,
        skip_if_no_models: None,
        chat_request_factory: Any,
    ) -> None:
        """Test streaming response uses SSE format."""
        request = chat_request_factory.streaming(
            message="Say 'hello world'",
            max_tokens=20,
        )

        async with client.stream("POST", "/v1/chat/completions", json=request) as response:
            assert response.status_code == 200

            # Check content type is SSE
            content_type = response.headers.get("content-type", "")
            assert "text/event-stream" in content_type

            # Read all content
            content = await response.aread()
            assert len(content) > 0

    @pytest.mark.requires_model
    async def test_streaming_chunks_have_data_prefix(
        self,
        client: httpx.AsyncClient,
        skip_if_no_models: None,
        chat_request_factory: Any,
        sse_parser: Any,
    ) -> None:
        """Test each chunk starts with 'data: ' prefix."""
        request = chat_request_factory.streaming(
            message="Count: 1, 2, 3",
            max_tokens=30,
        )

        async with client.stream("POST", "/v1/chat/completions", json=request) as response:
            content = (await response.aread()).decode("utf-8")

        # Parse and validate chunks
        chunks = sse_parser.parse_stream(content)
        assert len(chunks) > 0, "No chunks received"

        # Each non-DONE chunk should be a dict
        for chunk in chunks:
            if chunk != "[DONE]":
                assert isinstance(chunk, dict), f"Chunk is not dict: {chunk}"

    @pytest.mark.requires_model
    async def test_streaming_ends_with_done(
        self,
        client: httpx.AsyncClient,
        skip_if_no_models: None,
        chat_request_factory: Any,
        sse_parser: Any,
    ) -> None:
        """Test streaming response ends with 'data: [DONE]'.

        AC-21.2: SSE chunks end with `data: [DONE]\\n\\n`
        """
        request = chat_request_factory.streaming(
            message="Say 'done'",
            max_tokens=20,
        )

        async with client.stream("POST", "/v1/chat/completions", json=request) as response:
            content = (await response.aread()).decode("utf-8")

        # Check raw content ends with [DONE]
        assert "data: [DONE]" in content, "Stream should end with 'data: [DONE]'"

        # Parse and verify [DONE] is last chunk
        chunks = sse_parser.parse_stream(content)
        assert len(chunks) > 0
        assert chunks[-1] == "[DONE]", "Last chunk should be [DONE]"

    @pytest.mark.requires_model
    async def test_streaming_chunks_contain_delta(
        self,
        client: httpx.AsyncClient,
        skip_if_no_models: None,
        chat_request_factory: Any,
        sse_parser: Any,
        response_validator: Any,
    ) -> None:
        """Test streaming chunks contain delta field."""
        request = chat_request_factory.streaming(
            message="Hello",
            max_tokens=30,
        )

        async with client.stream("POST", "/v1/chat/completions", json=request) as response:
            content = (await response.aread()).decode("utf-8")

        chunks = sse_parser.parse_stream(content)

        # Validate each non-DONE chunk
        for chunk in chunks:
            if chunk != "[DONE]":
                response_validator.validate_streaming_chunk(chunk)

    @pytest.mark.requires_model
    async def test_streaming_extracts_content(
        self,
        client: httpx.AsyncClient,
        skip_if_no_models: None,
        chat_request_factory: Any,
        sse_parser: Any,
    ) -> None:
        """Test content can be extracted from stream."""
        request = chat_request_factory.streaming(
            message="Say exactly: 'Hello World'",
            max_tokens=30,
        )

        async with client.stream("POST", "/v1/chat/completions", json=request) as response:
            content = (await response.aread()).decode("utf-8")

        chunks = sse_parser.parse_stream(content)
        extracted = sse_parser.extract_content(chunks)

        # Should have extracted some content
        assert len(extracted) > 0, "No content extracted from stream"


# =============================================================================
# Streaming Chunk Structure Tests
# =============================================================================


class TestStreamingChunkStructure:
    """Test structure of streaming chunks."""

    @pytest.mark.requires_model
    async def test_first_chunk_has_role(
        self,
        client: httpx.AsyncClient,
        skip_if_no_models: None,
        chat_request_factory: Any,
        sse_parser: Any,
    ) -> None:
        """Test first chunk contains role in delta."""
        request = chat_request_factory.streaming(max_tokens=30)

        async with client.stream("POST", "/v1/chat/completions", json=request) as response:
            content = (await response.aread()).decode("utf-8")

        chunks = [c for c in sse_parser.parse_stream(content) if c != "[DONE]"]

        if chunks:
            first_chunk = chunks[0]
            delta = first_chunk.get("choices", [{}])[0].get("delta", {})
            # First chunk typically has role
            assert "role" in delta or "content" in delta

    @pytest.mark.requires_model
    async def test_chunks_have_consistent_id(
        self,
        client: httpx.AsyncClient,
        skip_if_no_models: None,
        chat_request_factory: Any,
        sse_parser: Any,
    ) -> None:
        """Test all chunks have the same ID."""
        request = chat_request_factory.streaming(max_tokens=50)

        async with client.stream("POST", "/v1/chat/completions", json=request) as response:
            content = (await response.aread()).decode("utf-8")

        chunks = [c for c in sse_parser.parse_stream(content) if isinstance(c, dict)]

        if len(chunks) > 1:
            first_id = chunks[0].get("id")
            for chunk in chunks[1:]:
                assert chunk.get("id") == first_id, "Chunk IDs should be consistent"

    @pytest.mark.requires_model
    async def test_last_data_chunk_has_finish_reason(
        self,
        client: httpx.AsyncClient,
        skip_if_no_models: None,
        chat_request_factory: Any,
        sse_parser: Any,
    ) -> None:
        """Test last data chunk has finish_reason."""
        request = chat_request_factory.streaming(max_tokens=30)

        async with client.stream("POST", "/v1/chat/completions", json=request) as response:
            content = (await response.aread()).decode("utf-8")

        chunks = [c for c in sse_parser.parse_stream(content) if isinstance(c, dict)]

        if chunks:
            last_chunk = chunks[-1]
            choice = last_chunk.get("choices", [{}])[0]
            assert "finish_reason" in choice, "Last chunk should have finish_reason"


# =============================================================================
# Streaming Error Handling Tests
# =============================================================================


class TestStreamingErrors:
    """Test error handling in streaming mode."""

    async def test_streaming_invalid_model(
        self,
        client: httpx.AsyncClient,
        skip_if_service_unavailable: None,
    ) -> None:
        """Test streaming with invalid model returns error."""
        request: dict[str, Any] = {
            "model": "nonexistent-model",
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": True,
            "max_tokens": 10,
        }

        response = await client.post("/v1/chat/completions", json=request)

        # Should return error status, not stream
        assert response.status_code in [400, 404]

    async def test_streaming_empty_messages(
        self,
        client: httpx.AsyncClient,
        skip_if_service_unavailable: None,
    ) -> None:
        """Test streaming with empty messages returns error."""
        request: dict[str, Any] = {
            "model": "llama-3.2-3b",
            "messages": [],
            "stream": True,
        }

        response = await client.post("/v1/chat/completions", json=request)

        assert response.status_code in [400, 422]


# =============================================================================
# Streaming Performance Tests
# =============================================================================


class TestStreamingPerformance:
    """Test streaming performance characteristics."""

    @pytest.mark.slow
    @pytest.mark.requires_model
    async def test_streaming_first_chunk_latency(
        self,
        client: httpx.AsyncClient,
        skip_if_no_models: None,
        chat_request_factory: Any,
    ) -> None:
        """Test time to first streaming chunk is reasonable."""
        import time

        request = chat_request_factory.streaming(
            message="Hello",
            max_tokens=50,
        )

        start_time = time.monotonic()
        first_chunk_time = None

        async with client.stream("POST", "/v1/chat/completions", json=request) as response:
            async for line in response.aiter_lines():
                if line.startswith("data: ") and line != "data: [DONE]":
                    first_chunk_time = time.monotonic() - start_time
                    break

        assert first_chunk_time is not None, "No chunks received"
        # First chunk should arrive within 30 seconds (model loading time)
        assert first_chunk_time < 30.0, f"First chunk took {first_chunk_time:.2f}s"

    @pytest.mark.slow
    @pytest.mark.requires_model
    async def test_streaming_receives_multiple_chunks(
        self,
        client: httpx.AsyncClient,
        skip_if_no_models: None,
        chat_request_factory: Any,
        sse_parser: Any,
    ) -> None:
        """Test streaming produces multiple chunks for longer responses."""
        request = chat_request_factory.streaming(
            message="Write a haiku about programming.",
            max_tokens=100,
        )

        async with client.stream("POST", "/v1/chat/completions", json=request) as response:
            content = (await response.aread()).decode("utf-8")

        chunks = [c for c in sse_parser.parse_stream(content) if isinstance(c, dict)]

        # Should have multiple content chunks
        assert len(chunks) >= 2, "Expected multiple streaming chunks"
