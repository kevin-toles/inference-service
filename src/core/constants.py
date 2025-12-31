"""Container path constants for inference-service.

This module centralizes all container filesystem paths to ensure consistency
across configuration, docker-compose, and runtime code.

All paths are for the CONTAINER environment, not the host.
Host paths are configured via environment variables in docker-compose.yml.

Container filesystem layout:
    /app/                   # Application root (WORKDIR in Dockerfile)
    /app/src/               # Application source code
    /app/config/            # Configuration files (models.yaml, presets.yaml)
    /app/models/            # Model files (mounted from host)

Usage:
    from src.core.constants import CONTAINER_MODELS_DIR, CONTAINER_CONFIG_DIR

Note: These are defaults. They can be overridden via environment variables:
    - INFERENCE_MODELS_DIR → overrides CONTAINER_MODELS_DIR
    - INFERENCE_CONFIG_DIR → overrides CONTAINER_CONFIG_DIR
"""

# =============================================================================
# Container Filesystem Paths
# =============================================================================
# These MUST match the volume mount targets in docker/docker-compose.yml

# Where model files (.gguf) are mounted in the container
# docker-compose: ${MODELS_PATH:-../models}:/app/models:ro
CONTAINER_MODELS_DIR = "/app/models"

# Where config files (models.yaml, presets.yaml) are mounted
# docker-compose: ../config:/app/config:ro
CONTAINER_CONFIG_DIR = "/app/config"


# =============================================================================
# Service Defaults
# =============================================================================

DEFAULT_SERVICE_NAME = "inference-service"
DEFAULT_PORT = 8085
DEFAULT_HOST = "0.0.0.0"
DEFAULT_LOG_LEVEL = "INFO"
DEFAULT_ENVIRONMENT = "development"
DEFAULT_BACKEND = "llamacpp"
DEFAULT_GPU_LAYERS = -1  # All layers on GPU/Metal
DEFAULT_ORCHESTRATION_MODE = "single"
