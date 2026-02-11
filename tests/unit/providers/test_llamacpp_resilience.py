"""Tests for C-7 resilience fixes in LlamaCppProvider.

Covers:
- Inference timeout (asyncio.wait_for wrapping)
- Auto-recovery from llama_decode -1 errors
- Qwen3-specific max_tokens cap
- Consecutive failure tracking
- IPv6/dual-stack host default
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.models.requests import ChatCompletionRequest, Message


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Fixtures
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@pytest.fixture
def mock_llama():
    """Create a mock Llama class that returns a configurable instance."""
    mock_class = MagicMock()
    mock_instance = MagicMock()
    mock_instance.tokenize.return_value = [1, 2, 3]
    mock_instance.create_chat_completion.return_value = {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 1700000000,
        "model": "test-model",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "ok"},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
    }
    mock_class.return_value = mock_instance
    return mock_class


@pytest.fixture
def simple_request():
    return ChatCompletionRequest(
        model="test-model",
        messages=[Message(role="user", content="Hello")],
    )


@pytest.fixture
def qwen3_request():
    """Request without explicit max_tokens (to test Qwen3 default cap)."""
    return ChatCompletionRequest(
        model="qwen3-8b",
        messages=[Message(role="user", content="Explain something")],
        max_tokens=None,
    )


def _make_provider(mock_llama, model_id="test-model", **kwargs):
    """Helper to create a loaded LlamaCppProvider with mocked Llama."""
    from src.providers.llamacpp import LlamaCppProvider

    with patch("src.providers.llamacpp.Llama", mock_llama):
        model_path = Path("/fake/model.gguf")
        with patch.object(Path, "exists", return_value=True):
            provider = LlamaCppProvider(
                model_path=model_path,
                model_id=model_id,
                context_length=2048,
                **kwargs,
            )
    # Manually set loaded state
    provider._model = mock_llama.return_value
    provider._is_loaded = True
    return provider


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Inference Timeout Tests
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestInferenceTimeout:
    """Test asyncio.wait_for() wrapping of inference calls."""

    async def test_generate_respects_timeout(self, mock_llama, simple_request):
        """Inference that hangs longer than timeout should raise."""
        from src.providers.llamacpp import LlamaCppInferenceError

        provider = _make_provider(mock_llama)
        provider._inference_timeout = 0.05  # 50ms

        # Make create_chat_completion block forever
        def slow_inference(*args, **kwargs):
            import time
            time.sleep(10)

        mock_llama.return_value.create_chat_completion.side_effect = slow_inference

        with pytest.raises(LlamaCppInferenceError, match="timed out"):
            await provider.generate(simple_request)

    async def test_generate_succeeds_within_timeout(self, mock_llama, simple_request):
        """Normal inference within timeout should succeed."""
        provider = _make_provider(mock_llama)
        provider._inference_timeout = 5.0

        result = await provider.generate(simple_request)
        assert result.choices[0].message.content == "ok"

    async def test_stream_init_respects_timeout(self, mock_llama, simple_request):
        """Stream initialization that hangs should raise."""
        from src.providers.llamacpp import LlamaCppInferenceError

        provider = _make_provider(mock_llama)
        provider._inference_timeout = 0.05

        def slow_init(*args, **kwargs):
            import time
            time.sleep(10)

        mock_llama.return_value.create_chat_completion.side_effect = slow_init

        with pytest.raises(LlamaCppInferenceError, match="timed out"):
            async for _ in provider.stream(simple_request):
                pass

    async def test_default_timeout_is_300s(self, mock_llama):
        """Default inference timeout should be 300 seconds."""
        from src.providers.llamacpp import DEFAULT_INFERENCE_TIMEOUT

        provider = _make_provider(mock_llama)
        assert provider._inference_timeout == DEFAULT_INFERENCE_TIMEOUT
        assert DEFAULT_INFERENCE_TIMEOUT == 300


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Auto-Recovery Tests
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestAutoRecovery:
    """Test auto-reload on llama_decode -1 errors."""

    async def test_llama_decode_failure_increments_counter(self, mock_llama, simple_request):
        """llama_decode -1 should increment consecutive failure count."""
        from src.providers.llamacpp import LlamaCppInferenceError

        provider = _make_provider(mock_llama)
        mock_llama.return_value.create_chat_completion.side_effect = RuntimeError(
            "llama_decode returned -1"
        )

        with pytest.raises(LlamaCppInferenceError):
            await provider.generate(simple_request)

        assert provider._consecutive_failures == 1

    async def test_auto_reload_after_max_consecutive_failures(
        self, mock_llama, simple_request
    ):
        """After N consecutive llama_decode failures, model should auto-reload."""
        from src.providers.llamacpp import LlamaCppInferenceError

        provider = _make_provider(mock_llama)
        provider._max_consecutive_failures = 2

        mock_llama.return_value.create_chat_completion.side_effect = RuntimeError(
            "llama_decode returned -1"
        )

        # First failure
        with pytest.raises(LlamaCppInferenceError):
            await provider.generate(simple_request)
        assert provider._consecutive_failures == 1

        # Second failure → triggers auto-reload
        with patch.object(provider, "load", new_callable=AsyncMock) as mock_load:
            with patch.object(provider, "unload", new_callable=AsyncMock) as mock_unload:
                with pytest.raises(LlamaCppInferenceError):
                    await provider.generate(simple_request)
                mock_unload.assert_called_once()
                mock_load.assert_called_once()
                # Counter should be reset after successful reload
                assert provider._consecutive_failures == 0

    async def test_success_resets_consecutive_failures(self, mock_llama, simple_request):
        """Successful inference should reset the failure counter."""
        provider = _make_provider(mock_llama)
        provider._consecutive_failures = 2

        await provider.generate(simple_request)
        assert provider._consecutive_failures == 0

    async def test_non_llama_decode_errors_increment_counter(
        self, mock_llama, simple_request
    ):
        """Non-llama_decode errors should still increment the counter but not trigger reload."""
        from src.providers.llamacpp import LlamaCppInferenceError

        provider = _make_provider(mock_llama)
        provider._max_consecutive_failures = 2
        mock_llama.return_value.create_chat_completion.side_effect = RuntimeError(
            "some other error"
        )

        with pytest.raises(LlamaCppInferenceError):
            await provider.generate(simple_request)
        assert provider._consecutive_failures == 1

        # Second failure — not llama_decode, so no reload
        with patch.object(provider, "load", new_callable=AsyncMock) as mock_load:
            with pytest.raises(LlamaCppInferenceError):
                await provider.generate(simple_request)
            mock_load.assert_not_called()
            assert provider._consecutive_failures == 2


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Qwen3 Max Tokens Cap Tests
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestQwen3MaxTokensCap:
    """Test Qwen3-specific max_tokens default to reduce <think> inflation."""

    async def test_qwen3_default_max_tokens_is_2048(self, mock_llama, qwen3_request):
        """Qwen3 models without explicit max_tokens should default to 2048."""
        provider = _make_provider(mock_llama, model_id="qwen3-8b")
        await provider.generate(qwen3_request)

        call_kwargs = mock_llama.return_value.create_chat_completion.call_args
        assert call_kwargs.kwargs.get("max_tokens") == 2048

    async def test_qwen3_coder_also_gets_cap(self, mock_llama, qwen3_request):
        """Qwen3-coder-30b should also get the 2048 cap."""
        provider = _make_provider(mock_llama, model_id="qwen3-coder-30b")
        await provider.generate(qwen3_request)

        call_kwargs = mock_llama.return_value.create_chat_completion.call_args
        assert call_kwargs.kwargs.get("max_tokens") == 2048

    async def test_qwen3_explicit_max_tokens_is_respected(self, mock_llama):
        """If caller specifies max_tokens, it should override the Qwen3 cap."""
        request = ChatCompletionRequest(
            model="qwen3-8b",
            messages=[Message(role="user", content="Hello")],
            max_tokens=512,
        )
        provider = _make_provider(mock_llama, model_id="qwen3-8b")
        await provider.generate(request)

        call_kwargs = mock_llama.return_value.create_chat_completion.call_args
        assert call_kwargs.kwargs.get("max_tokens") == 512

    async def test_non_qwen3_model_defaults_to_4096(self, mock_llama, simple_request):
        """Non-Qwen3 models should still default to 4096 (C-6 fix)."""
        provider = _make_provider(mock_llama, model_id="phi-4")
        await provider.generate(simple_request)

        call_kwargs = mock_llama.return_value.create_chat_completion.call_args
        assert call_kwargs.kwargs.get("max_tokens") == 4096


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Config Tests
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestC7ConfigDefaults:
    """Verify C-7 configuration defaults."""

    def test_default_host_is_dual_stack(self):
        """DEFAULT_HOST should be :: for dual-stack IPv4+IPv6."""
        from src.core.constants import DEFAULT_HOST
        assert DEFAULT_HOST == "::"

    def test_inference_timeout_default(self):
        """Settings should have inference_timeout = 300."""
        import os
        # Clear any env overrides
        env_backup = os.environ.get("INFERENCE_INFERENCE_TIMEOUT")
        os.environ.pop("INFERENCE_INFERENCE_TIMEOUT", None)
        try:
            from src.core.config import Settings
            s = Settings(skip_path_validation=True)
            assert s.inference_timeout == 300
        finally:
            if env_backup is not None:
                os.environ["INFERENCE_INFERENCE_TIMEOUT"] = env_backup

    def test_max_consecutive_failures_default(self):
        """Settings should have inference_max_consecutive_failures = 3."""
        from src.core.config import Settings
        s = Settings(skip_path_validation=True)
        assert s.inference_max_consecutive_failures == 3

    def test_consecutive_failure_constants(self):
        """Provider constants should match expected values."""
        from src.providers.llamacpp import (
            DEFAULT_INFERENCE_TIMEOUT,
            DEFAULT_MAX_CONSECUTIVE_FAILURES,
            QWEN3_DEFAULT_MAX_TOKENS,
        )
        assert DEFAULT_INFERENCE_TIMEOUT == 300
        assert DEFAULT_MAX_CONSECUTIVE_FAILURES == 3
        assert QWEN3_DEFAULT_MAX_TOKENS == 2048
