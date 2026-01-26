"""
Observability Package - OBS-11: Distributed Tracing Propagation

This package provides OpenTelemetry tracing for inference-service.
Traces are exported to Jaeger via OTLP gRPC.

Reference: WBS_B5_B1_REMAINING_WORK.md - OBS-11
"""

from src.observability.tracing import (
    setup_tracing,
    TracingMiddleware,
    get_tracer,
    get_current_trace_id,
    get_current_span_id,
    inject_trace_context,
    extract_trace_context,
)

__all__ = [
    "setup_tracing",
    "TracingMiddleware",
    "get_tracer",
    "get_current_trace_id",
    "get_current_span_id",
    "inject_trace_context",
    "extract_trace_context",
]
