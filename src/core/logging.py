"""Structured logging module for inference-service.

Provides JSON-formatted structured logging using structlog.

Patterns applied:
- Singleton _configured flag prevents reconfiguration - Issue #16
- configure_logging() called ONCE at startup
- Underscore-prefix for unused structlog params - AP-4.3
- JSON output via JSONRenderer
- Correlation ID support via contextvars

Reference: WBS-INF2 AC-2.2
Exit Criteria: Log output is valid JSON with timestamp, level, message
"""

import contextvars
import sys
from typing import Any, TextIO

import structlog
from structlog.types import EventDict


# =============================================================================
# Singleton Configuration State
# =============================================================================
_configured: bool = False


# =============================================================================
# Correlation ID Context (for request tracing)
# =============================================================================
_correlation_id_var: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "correlation_id", default=None
)


def set_correlation_id(correlation_id: str) -> None:
    """Set correlation ID for current async context.

    Args:
        correlation_id: Unique request identifier for tracing.
    """
    _correlation_id_var.set(correlation_id)


def get_correlation_id() -> str | None:
    """Get current correlation ID.

    Returns:
        Correlation ID if set, None otherwise.
    """
    return _correlation_id_var.get()


# =============================================================================
# Custom Processors
# =============================================================================
def add_correlation_id(
    _logger: object, _method_name: str, event_dict: EventDict
) -> EventDict:
    """Add correlation ID to log event if set.

    Args:
        _logger: Logger instance (unused - required by structlog interface).
        _method_name: Method name (unused - underscore prefix per AP-4.3).
        event_dict: Event dictionary to process.

    Returns:
        Event dictionary with correlation_id added if set.
    """
    correlation_id = get_correlation_id()
    if correlation_id is not None:
        event_dict["correlation_id"] = correlation_id
    return event_dict


def _level_to_int(level: str) -> int:
    """Convert log level string to integer.

    Args:
        level: Log level name (DEBUG, INFO, WARNING, ERROR, CRITICAL).

    Returns:
        Integer log level for structlog filtering.
    """
    levels = {
        "DEBUG": 10,
        "INFO": 20,
        "WARNING": 30,
        "ERROR": 40,
        "CRITICAL": 50,
    }
    return levels.get(level.upper(), 20)


# =============================================================================
# Singleton Configuration
# =============================================================================
def configure_logging(
    level: str = "INFO",
    stream: TextIO | None = None,
    force: bool = False,
) -> None:
    """Configure structlog ONCE at application startup.

    Subsequent calls are no-ops unless force=True (for testing).
    This follows Issue #16 pattern - configure_logging() singleton.

    Args:
        level: Minimum log level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        stream: Output stream. Defaults to sys.stdout.
        force: Force reconfiguration (for testing only).
    """
    global _configured

    if _configured and not force:
        return

    processors: list[structlog.types.Processor] = [
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", key="timestamp"),
        add_correlation_id,
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.JSONRenderer(),
    ]

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(_level_to_int(level)),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(file=stream or sys.stdout),
        cache_logger_on_first_use=False,
    )

    _configured = True


def reset_logging() -> None:
    """Reset configuration state for test isolation.

    Only use in tests to allow reconfiguration between tests.
    """
    global _configured
    _configured = False


def get_logger(name: str) -> Any:
    """Get configured logger by name.

    Auto-configures with defaults if not already configured.
    This ensures logging always works even if configure_logging()
    wasn't called explicitly.

    Args:
        name: Logger name (typically __name__ of the module).

    Returns:
        Configured structlog BoundLogger instance.
    """
    configure_logging()  # No-op if already configured
    return structlog.get_logger().bind(logger=name)
