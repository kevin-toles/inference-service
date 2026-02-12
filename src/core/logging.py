"""Structured logging module for inference-service.

Provides JSON-formatted structured logging using structlog.

WBS-LOG0: Structured Logging Implementation
- AC-LOG0.1: JSONFormatter with timestamp, level, service, correlation_id, module, message
- AC-LOG0.2: RotatingFileHandler writing to /var/log/inference-service/app.log
- AC-LOG0.3: CorrelationIdFilter for X-Request-ID propagation
- AC-LOG0.4: Log level configurable via INFERENCE_SERVICE_LOG_LEVEL env var

Patterns applied:
- Singleton _configured flag prevents reconfiguration - Issue #16
- configure_logging() called ONCE at startup
- Underscore-prefix for unused structlog params - AP-4.3
- JSON output via JSONRenderer
- Correlation ID support via contextvars

Reference: WBS-INF2 AC-2.2, WBS-LOG0
Exit Criteria: Log output is valid JSON with timestamp, level, message
"""

import contextvars
import json
import logging
import os
import sys
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, TextIO

import structlog
from structlog.types import EventDict


# =============================================================================
# Singleton Configuration State
# =============================================================================
_configured: bool = False

# Default log file path for Linux deployments (S1192)
_DEFAULT_LINUX_LOG_PATH = "/var/log/inference-service/app.log"


# =============================================================================
# Correlation ID Context (AC-LOG0.3)
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


def clear_correlation_id() -> None:
    """Clear the correlation ID from context."""
    _correlation_id_var.set(None)


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
    else:
        event_dict["correlation_id"] = "-"
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
def _get_default_log_path() -> str:
    """Get platform-appropriate default log file path.
    
    Returns:
        macOS: ~/Library/Logs/ai-platform/inference-service/app.log
        Linux: /var/log/inference-service/app.log
    """
    if sys.platform == "darwin":
        home = Path.home()
        return str(home / "Library" / "Logs" / "ai-platform" / "inference-service" / "app.log")
    return _DEFAULT_LINUX_LOG_PATH


def configure_logging(
    level: str = "INFO",
    stream: TextIO | None = None,
    force: bool = False,
    log_file_path: str | None = None,
    enable_file_logging: bool | None = None,
) -> None:
    """Configure structlog ONCE at application startup.

    Subsequent calls are no-ops unless force=True (for testing).
    This follows Issue #16 pattern - configure_logging() singleton.

    Args:
        level: Minimum log level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        stream: Output stream. Defaults to sys.stdout.
        force: Force reconfiguration (for testing only).
        log_file_path: Path for log file. Defaults to platform-appropriate location.
        enable_file_logging: Whether to enable file logging. 
            Defaults from INFERENCE_ENABLE_FILE_LOGGING env var.
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

    # Setup file logging if enabled
    if enable_file_logging is None:
        enable_file_logging = os.environ.get("INFERENCE_ENABLE_FILE_LOGGING", "true").lower() in ("true", "1", "yes")
    
    if enable_file_logging:
        if log_file_path is None:
            log_file_path = os.environ.get("INFERENCE_LOG_FILE_PATH") or _get_default_log_path()
        
        try:
            file_handler = create_file_handler(log_file_path, "inference-service")
            file_handler.setLevel(_level_to_int(level))
            # Add to root logger to capture all logs
            root_logger = logging.getLogger()
            root_logger.setLevel(_level_to_int(level))
            root_logger.addHandler(file_handler)
        except OSError as e:
            # Log warning to stdout if file logging fails
            print(f'{{"timestamp": "{datetime.now(timezone.utc).isoformat()}", "level": "WARNING", "service": "inference-service", "message": "File logging disabled: {e}"}}', file=sys.stderr)

    _configured = True


def reset_logging() -> None:
    """Reset configuration state for test isolation.

    Only use in tests to allow reconfiguration between tests.
    """
    global _configured
    _configured = False


# =============================================================================
# WBS-LOG0: Standard Library Logging Integration
# =============================================================================
class JSONFormatter(logging.Formatter):
    """JSON log formatter with standard fields (AC-LOG0.1)."""
    
    def __init__(self, service_name: str = "inference-service", **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.service_name = service_name
    
    def format(self, record: logging.LogRecord) -> str:
        correlation_id = getattr(record, "correlation_id", "-")
        
        log_data = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "service": self.service_name,
            "correlation_id": correlation_id,
            "module": record.module,
            "message": record.getMessage(),
        }
        
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
        
        return json.dumps(log_data, ensure_ascii=False)


class CorrelationIdFilter(logging.Filter):
    """Filter that adds correlation ID to log records (AC-LOG0.3)."""
    
    def filter(self, record: logging.LogRecord) -> bool:
        correlation_id = get_correlation_id()
        record.correlation_id = correlation_id if correlation_id else "-"
        return True


def get_log_level_from_env(service_prefix: str = "INFERENCE_SERVICE") -> int:
    """Get log level from INFERENCE_SERVICE_LOG_LEVEL env var (AC-LOG0.4)."""
    env_var = f"{service_prefix}_LOG_LEVEL"
    level_str = os.environ.get(env_var, "INFO").upper()
    level = getattr(logging, level_str, None)
    return level if isinstance(level, int) else logging.INFO


def create_file_handler(
    log_file_path: str = _DEFAULT_LINUX_LOG_PATH,
    service_name: str = "inference-service",
    max_bytes: int = 10_485_760,
    backup_count: int = 5,
) -> RotatingFileHandler:
    """Create a rotating file handler for JSON logs (AC-LOG0.2)."""
    log_dir = Path(log_file_path).parent
    log_dir.mkdir(parents=True, exist_ok=True)
    
    handler = RotatingFileHandler(
        filename=log_file_path,
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8",
    )
    handler.setFormatter(JSONFormatter(service_name=service_name))
    handler.addFilter(CorrelationIdFilter())
    return handler


def setup_structured_logging(
    service_name: str = "inference-service",
    log_file_path: str | None = _DEFAULT_LINUX_LOG_PATH,
    log_level: int | None = None,
) -> logging.Logger:
    """Set up structured logging with file handler (WBS-LOG0)."""
    if log_level is None:
        log_level = get_log_level_from_env()
    
    logger = logging.getLogger(service_name)
    logger.setLevel(log_level)
    logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(JSONFormatter(service_name=service_name))
    console_handler.addFilter(CorrelationIdFilter())
    logger.addHandler(console_handler)
    
    # File handler
    if log_file_path:
        try:
            file_handler = create_file_handler(log_file_path, service_name)
            file_handler.setLevel(log_level)
            logger.addHandler(file_handler)
        except PermissionError:
            logger.warning(f"Cannot write to {log_file_path}, file logging disabled")
    
    logger.propagate = False
    return logger


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
