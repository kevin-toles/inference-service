# ==============================================================================
# Inference Service - Production Dockerfile
# ==============================================================================
# Multi-stage build for optimized production image with llama-cpp-python
# Supports Apple Metal acceleration on macOS and CPU on Linux
# Reference: docs/ARCHITECTURE.md → Docker Health Check
# ==============================================================================

# ==============================================================================
# Stage 1: Builder
# ==============================================================================
FROM python:3.11-slim-bookworm AS builder

# Prevent Python from writing bytecode and buffering stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install build dependencies for llama-cpp-python
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    curl \
    git \
    ninja-build \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python dependencies
WORKDIR /build
COPY requirements.txt .

# Build llama-cpp-python without hardware acceleration (portable)
# For optimized builds, use platform-specific Dockerfiles
ENV CMAKE_ARGS="-DGGML_NATIVE=OFF"
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# ==============================================================================
# Stage 2: Production Runtime
# ==============================================================================
FROM python:3.11-slim-bookworm AS production

# Labels (OCI standard)
LABEL org.opencontainers.image.title="inference-service" \
      org.opencontainers.image.description="Self-hosted LLM inference service with llama-cpp-python" \
      org.opencontainers.image.version="0.1.0" \
      org.opencontainers.image.vendor="kevin-toles" \
      org.opencontainers.image.source="https://github.com/kevin-toles/inference-service"

# Security: Run as non-root user
RUN groupadd --gid 1000 appgroup && \
    useradd --uid 1000 --gid appgroup --shell /bin/bash --create-home appuser

# Runtime environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src \
    PATH="/opt/venv/bin:$PATH" \
    # Application defaults (override via docker-compose or Kubernetes)
    INFERENCE_PORT=8085 \
    INFERENCE_LOG_LEVEL=INFO \
    INFERENCE_WORKERS=1

# Install runtime dependencies
# libgomp1 is required by llama-cpp-python for OpenMP parallelism
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv

# ==============================================================================
# Stage 3: Application Setup
# ==============================================================================

# Set working directory
WORKDIR /app

# Copy application code with correct ownership (read-only for security)
COPY --chown=appuser:appgroup src/ ./src/
COPY --chown=appuser:appgroup config/ ./config/

# Set permissions and create models directory (merged for layer optimization)
RUN chmod -R a-w ./src/ ./config/ && \
    mkdir -p /app/models && \
    chown appuser:appgroup /app/models

# Switch to non-root user
USER appuser

# ==============================================================================
# Stage 4: Container Configuration
# ==============================================================================

# Health check using /health endpoint
# Reference: docs/ARCHITECTURE.md → Docker Health Check
HEALTHCHECK --interval=60s --timeout=30s --start-period=120s --retries=3 \
    CMD curl -f "http://localhost:${INFERENCE_PORT}/health" || exit 1

# Expose port (documentation)
EXPOSE 8085

# Default command - single worker for model inference
# llama-cpp-python is not multi-process safe, use 1 worker
CMD ["sh", "-c", "uvicorn src.main:app --host 0.0.0.0 --port \"${INFERENCE_PORT:-8085}\" --workers \"${INFERENCE_WORKERS:-1}\""]
