"""FastAPI application entrypoint for inference-service.

Patterns applied:
- asynccontextmanager lifespan (modern FastAPI pattern, not deprecated @app.on_event)
- configure_logging() called ONCE in lifespan startup
- Health endpoints for K8s probes (/health, /health/ready)
- Docs disabled in production

Reference: WBS-INF2 AC-2.3, WBS-INF7
"""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from src.api.error_handlers import register_exception_handlers
from src.api.routes.completions import router as completions_router
from src.api.routes.health import router as health_router
from src.api.routes.models import router as models_router
from src.core.config import get_settings
from src.core.logging import configure_logging, get_logger


# =============================================================================
# Application Metadata
# =============================================================================
APP_NAME = "inference-service"
APP_DESCRIPTION = "Local LLM inference service with multi-model orchestration"
APP_VERSION = "0.1.0"


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

    yield

    # =========================================================================
    # SHUTDOWN
    # =========================================================================
    logger.info("Application shutting down", service=APP_NAME)
    app.state.initialized = False


# =============================================================================
# FastAPI Application Instance
# =============================================================================
settings = get_settings()

app = FastAPI(
    title=APP_NAME,
    description=APP_DESCRIPTION,
    version=APP_VERSION,
    docs_url="/docs" if settings.environment != "production" else None,
    redoc_url="/redoc" if settings.environment != "production" else None,
    lifespan=lifespan,
)


# =============================================================================
# Register Routers (WBS-INF7, WBS-INF8, WBS-INF9)
# =============================================================================
app.include_router(health_router)
app.include_router(models_router, prefix="/v1")
app.include_router(completions_router, prefix="/v1")


# =============================================================================
# Register Exception Handlers (WBS-INF18)
# =============================================================================
register_exception_handlers(app)
