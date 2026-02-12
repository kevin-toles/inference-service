"""FastAPI application entrypoint for inference-service.

Patterns applied:
- asynccontextmanager lifespan (modern FastAPI pattern, not deprecated @app.on_event)
- configure_logging() called ONCE in lifespan startup
- Health endpoints for K8s probes (/health, /health/ready)
- Docs disabled in production
- Auto-load default preset on startup (INFERENCE_DEFAULT_PRESET env var)
- OBS-11: Distributed tracing propagation

Reference: WBS-INF2 AC-2.3, WBS-INF7
"""

import os
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from src.api.error_handlers import register_exception_handlers
from src.api.routes.completions import router as completions_router
from src.api.routes.health import router as health_router
from src.api.routes.models import router as models_router
from src.core.config import get_settings
from src.core.logging import configure_logging, get_logger
from src.services.model_manager import get_model_manager
from src.services.config_publisher import initialize_publisher, shutdown_publisher
from src.services.audit_client import init_audit_client, get_audit_client
from src.services.queue_manager import QueueManager, QueueStrategy

# OBS-11: Distributed tracing propagation
from src.observability import setup_tracing, TracingMiddleware


# =============================================================================
# Application Metadata
# =============================================================================
APP_NAME = "inference-service"
APP_DESCRIPTION = "Local LLM inference service with multi-model orchestration"
APP_VERSION = "0.1.0"


# =============================================================================
# Lifespan Init Helpers (S3776: extracted to reduce cognitive complexity)
# =============================================================================


async def _init_redis_publisher(settings: Any, logger: Any) -> None:
    """Initialize Redis config publisher if enabled."""
    if not settings.redis_enabled:
        return
    try:
        publisher = await initialize_publisher(settings.redis_url)
        if publisher.is_connected:
            logger.info(
                "Redis config publisher initialized",
                redis_url=settings.redis_url,
            )
        else:
            logger.warning(
                "Redis unavailable - config changes will not be published",
                redis_url=settings.redis_url,
            )
    except Exception as e:
        logger.warning(
            "Failed to initialize Redis publisher",
            error=str(e),
        )


async def _init_neo4j_audit(logger: Any) -> None:
    """Initialize Neo4j audit client."""
    try:
        audit_client = await init_audit_client()
        if audit_client and audit_client.is_connected:
            logger.info(
                "Neo4j audit client initialized",
                uri=audit_client._config.uri,
            )
        else:
            logger.info(
                "Neo4j audit logging disabled or unavailable",
            )
    except Exception as e:
        logger.warning(
            "Failed to initialize Neo4j audit client",
            error=str(e),
        )


async def _init_model_manager(app: FastAPI, settings: Any, logger: Any) -> None:
    """Initialize model manager and optionally load default preset."""
    model_manager = get_model_manager()
    app.state.model_manager = model_manager

    if not settings.default_preset:
        logger.info(
            "No default preset configured - set INFERENCE_DEFAULT_PRESET to auto-load models"
        )
        app.state.current_preset = None
        app.state.models_loaded = []
        return

    logger.info(
        "Auto-loading default preset on startup",
        preset=settings.default_preset,
    )
    try:
        result = await model_manager.load_preset(settings.default_preset)
        logger.info(
            "Default preset loaded successfully",
            preset=result.preset_id,
            models=result.models_loaded,
            memory_gb=result.total_memory_gb,
            orchestration_mode=result.orchestration_mode,
        )
        app.state.current_preset = result.preset_id
        app.state.models_loaded = result.models_loaded
    except Exception as e:
        logger.error(
            "Failed to load default preset - service will start without models",
            preset=settings.default_preset,
            error=str(e),
        )
        app.state.current_preset = None
        app.state.models_loaded = []


# =============================================================================
# Lifespan Context Manager (Modern FastAPI Pattern)
# =============================================================================
@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan - startup and shutdown events.

    This is the modern pattern for FastAPI (vs deprecated @app.on_event).
    Logging is configured ONCE here per Issue #16.

    Args:
        app: FastAPI application instance.

    Yields:
        None after startup, before shutdown.
    """
    # =========================================================================
    # STARTUP
    # =========================================================================
    settings = get_settings()

    # Configure logging ONCE (Issue #16 - singleton pattern)
    configure_logging(level=settings.log_level)
    logger = get_logger(__name__)

    logger.info(
        "Application starting",
        service=APP_NAME,
        version=APP_VERSION,
        environment=settings.environment,
        port=settings.port,
    )

    # Initialize app state
    app.state.initialized = True
    app.state.environment = settings.environment
    app.state.service_name = settings.service_name

    # Initialize subsystems via extracted helpers
    await _init_redis_publisher(settings, logger)
    await _init_neo4j_audit(logger)
    await _init_model_manager(app, settings, logger)

    # H-1: Initialize QueueManager for GPU concurrency control
    queue_strategy = (
        QueueStrategy.PRIORITY
        if settings.queue_strategy == "priority"
        else QueueStrategy.FIFO
    )
    queue_manager = QueueManager(
        max_concurrent=settings.max_concurrent,
        strategy=queue_strategy,
        reject_when_full=False,
        max_size=settings.queue_max_size,
    )
    app.state.queue_manager = queue_manager
    logger.info(
        "QueueManager initialized",
        max_concurrent=settings.max_concurrent,
        strategy=settings.queue_strategy,
        max_size=settings.queue_max_size,
    )

    yield

    # =========================================================================
    # SHUTDOWN
    # =========================================================================
    logger.info("Application shutting down", service=APP_NAME)

    # H-1: Gracefully shut down QueueManager (drain active, clear pending)
    if hasattr(app.state, "queue_manager") and app.state.queue_manager is not None:
        shutdown_stats = await app.state.queue_manager.shutdown(timeout=30.0)
        logger.info(
            "QueueManager shut down",
            active_drained=shutdown_stats["active_drained"],
            pending_cleared=shutdown_stats["pending_cleared"],
            timed_out=shutdown_stats["timed_out"],
        )

    # LLM Operations Mesh - Phase 5: Shutdown Neo4j audit client
    audit_client = get_audit_client()
    if audit_client:
        await audit_client.close()
    
    # LLM Operations Mesh - Phase 2: Shutdown Redis publisher
    await shutdown_publisher()
    
    app.state.initialized = False


# =============================================================================
# FastAPI Application Instance
# =============================================================================
settings = get_settings()

# OBS-11: Initialize OpenTelemetry tracing
otlp_endpoint = os.getenv("OTLP_ENDPOINT", "http://localhost:4317")
tracing_enabled = os.getenv("TRACING_ENABLED", "true").lower() == "true"
if tracing_enabled:
    setup_tracing(service_name="inference-service", otlp_endpoint=otlp_endpoint)

app = FastAPI(
    title=APP_NAME,
    description=APP_DESCRIPTION,
    version=APP_VERSION,
    docs_url="/docs" if settings.environment != "production" else None,
    redoc_url="/redoc" if settings.environment != "production" else None,
    lifespan=lifespan,
)

# OBS-11: Add TracingMiddleware for distributed tracing
if tracing_enabled:
    app.add_middleware(
        TracingMiddleware,
        exclude_paths=["/health", "/health/ready", "/health/live", "/metrics"],
    )


# =============================================================================
# Register Routers (WBS-INF7, WBS-INF8, WBS-INF9, WBS-IMG6)
# =============================================================================
app.include_router(health_router)
app.include_router(models_router, prefix="/v1")
app.include_router(completions_router, prefix="/v1")


# =============================================================================
# Register Exception Handlers (WBS-INF18)
# =============================================================================
register_exception_handlers(app)
