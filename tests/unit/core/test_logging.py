"""Tests for structured logging module.

TDD RED Phase: These tests define the expected behavior of logging configuration.
Reference: WBS-INF2 Exit Criteria, CODING_PATTERNS_ANALYSIS.md Issue #16

Tests verify:
- AC-2.2: Structured logging with JSON format
- Exit Criteria: Log output is valid JSON with timestamp, level, message
"""

import json
from io import StringIO

import pytest


class TestConfigureLogging:
    """Test configure_logging() function."""

    def test_configure_logging_exists(self) -> None:
        """configure_logging function is importable."""
        from src.core.logging import configure_logging

        assert callable(configure_logging)

    def test_configure_logging_with_level(self) -> None:
        """configure_logging accepts level parameter."""
        from src.core.logging import configure_logging, reset_logging

        reset_logging()
        configure_logging(level="DEBUG")
        # Should not raise

    def test_configure_logging_only_runs_once(self) -> None:
        """configure_logging is idempotent - only configures once."""
        from src.core.logging import configure_logging, reset_logging

        reset_logging()
        configure_logging(level="DEBUG")
        # Second call should be no-op
        configure_logging(level="ERROR")  # Would change level if it ran again

    def test_configure_logging_force_reconfigures(self) -> None:
        """configure_logging with force=True reconfigures."""
        from src.core.logging import configure_logging, reset_logging

        reset_logging()
        configure_logging(level="DEBUG")
        configure_logging(level="ERROR", force=True)  # Should reconfigure


class TestResetLogging:
    """Test reset_logging() function for test isolation."""

    def test_reset_logging_exists(self) -> None:
        """reset_logging function is importable."""
        from src.core.logging import reset_logging

        assert callable(reset_logging)

    def test_reset_allows_reconfiguration(self) -> None:
        """After reset, configure_logging runs again."""
        from src.core.logging import configure_logging, reset_logging

        configure_logging(level="INFO")
        reset_logging()
        # Should be able to reconfigure now
        configure_logging(level="DEBUG")


class TestGetLogger:
    """Test get_logger() function."""

    def test_get_logger_returns_bound_logger(self) -> None:
        """get_logger returns a structlog BoundLogger."""
        from src.core.logging import get_logger, reset_logging

        reset_logging()
        logger = get_logger("test.module")
        assert logger is not None
        assert hasattr(logger, "info")
        assert hasattr(logger, "error")
        assert hasattr(logger, "debug")
        assert hasattr(logger, "warning")

    def test_get_logger_binds_name(self) -> None:
        """get_logger binds the logger name."""
        from src.core.logging import get_logger, reset_logging

        reset_logging()
        logger = get_logger("my.module.name")
        # Logger should have the name bound
        assert logger is not None

    def test_get_logger_auto_configures(self) -> None:
        """get_logger auto-configures if not already configured."""
        from src.core.logging import get_logger, reset_logging

        reset_logging()
        # Don't call configure_logging first
        logger = get_logger("auto.config.test")
        assert logger is not None


class TestJSONOutput:
    """Test that log output is valid JSON."""

    def test_log_output_is_valid_json(self) -> None:
        """Log output is parseable JSON.

        Exit Criteria: Log output is valid JSON with timestamp, level, message.
        """
        from src.core.logging import configure_logging, get_logger, reset_logging

        reset_logging()
        stream = StringIO()
        configure_logging(level="INFO", stream=stream, force=True)

        logger = get_logger("json.test")
        logger.info("Test message")

        output = stream.getvalue()
        assert output, "Expected log output"

        # Parse as JSON
        log_record = json.loads(output.strip())
        assert isinstance(log_record, dict)

    def test_log_output_has_timestamp(self) -> None:
        """Log output contains timestamp field."""
        from src.core.logging import configure_logging, get_logger, reset_logging

        reset_logging()
        stream = StringIO()
        configure_logging(level="INFO", stream=stream, force=True)

        logger = get_logger("timestamp.test")
        logger.info("Test message")

        log_record = json.loads(stream.getvalue().strip())
        assert "timestamp" in log_record

    def test_log_output_has_level(self) -> None:
        """Log output contains level field."""
        from src.core.logging import configure_logging, get_logger, reset_logging

        reset_logging()
        stream = StringIO()
        configure_logging(level="INFO", stream=stream, force=True)

        logger = get_logger("level.test")
        logger.info("Test message")

        log_record = json.loads(stream.getvalue().strip())
        assert "level" in log_record
        assert log_record["level"] == "info"

    def test_log_output_has_event_message(self) -> None:
        """Log output contains event (message) field."""
        from src.core.logging import configure_logging, get_logger, reset_logging

        reset_logging()
        stream = StringIO()
        configure_logging(level="INFO", stream=stream, force=True)

        logger = get_logger("message.test")
        logger.info("My test message")

        log_record = json.loads(stream.getvalue().strip())
        assert "event" in log_record
        assert log_record["event"] == "My test message"


class TestLogLevels:
    """Test log level filtering."""

    def test_debug_not_logged_at_info_level(self) -> None:
        """DEBUG messages filtered when level is INFO."""
        from src.core.logging import configure_logging, get_logger, reset_logging

        reset_logging()
        stream = StringIO()
        configure_logging(level="INFO", stream=stream, force=True)

        logger = get_logger("filter.test")
        logger.debug("Debug message")

        output = stream.getvalue()
        assert output == ""  # Should be filtered

    def test_info_logged_at_info_level(self) -> None:
        """INFO messages logged when level is INFO."""
        from src.core.logging import configure_logging, get_logger, reset_logging

        reset_logging()
        stream = StringIO()
        configure_logging(level="INFO", stream=stream, force=True)

        logger = get_logger("info.test")
        logger.info("Info message")

        output = stream.getvalue()
        assert output != ""

    def test_error_logged_at_info_level(self) -> None:
        """ERROR messages logged when level is INFO."""
        from src.core.logging import configure_logging, get_logger, reset_logging

        reset_logging()
        stream = StringIO()
        configure_logging(level="INFO", stream=stream, force=True)

        logger = get_logger("error.test")
        logger.error("Error message")

        output = stream.getvalue()
        assert output != ""
        log_record = json.loads(output.strip())
        assert log_record["level"] == "error"


class TestCorrelationId:
    """Test correlation ID context support."""

    def test_set_correlation_id_exists(self) -> None:
        """set_correlation_id function is importable."""
        from src.core.logging import set_correlation_id

        assert callable(set_correlation_id)

    def test_get_correlation_id_exists(self) -> None:
        """get_correlation_id function is importable."""
        from src.core.logging import get_correlation_id

        assert callable(get_correlation_id)

    def test_correlation_id_roundtrip(self) -> None:
        """set/get correlation ID works correctly."""
        from src.core.logging import get_correlation_id, set_correlation_id

        test_id = "req-12345-abcde"
        set_correlation_id(test_id)
        assert get_correlation_id() == test_id

    def test_correlation_id_in_log_output(self) -> None:
        """Correlation ID appears in log output when set."""
        from src.core.logging import (
            configure_logging,
            get_logger,
            reset_logging,
            set_correlation_id,
        )

        reset_logging()
        stream = StringIO()
        configure_logging(level="INFO", stream=stream, force=True)

        set_correlation_id("corr-98765")
        logger = get_logger("corr.test")
        logger.info("Test with correlation")

        log_record = json.loads(stream.getvalue().strip())
        assert "correlation_id" in log_record
        assert log_record["correlation_id"] == "corr-98765"
