"""
OpenTelemetry Tracing Module - OBS-11: Distributed Tracing Propagation

This module provides distributed tracing for inference-service.
Trace context is propagated from incoming requests (llm-gateway, ai-agents)
and recorded in Jaeger.

Reference: WBS_B5_B1_REMAINING_WORK.md - OBS-11
"""

from typing import Any, Callable, Optional

from opentelemetry import trace
from opentelemetry.context import Context
from opentelemetry.propagate import extract, inject
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from opentelemetry.trace import SpanKind, Tracer

# Global tracer provider reference
_tracer_provider: Optional[TracerProvider] = None


def setup_tracing(
    service_name: str = "inference-service",
    otlp_endpoint: Optional[str] = None,
) -> TracerProvider:
    """
    Configure OpenTelemetry TracerProvider.

    Args:
        service_name: Name of the service for resource identification
        otlp_endpoint: Optional OTLP exporter endpoint (e.g., http://localhost:4317)

    Returns:
        Configured TracerProvider
    """
    global _tracer_provider

    resource = Resource.create({SERVICE_NAME: service_name})
    provider = TracerProvider(resource=resource)

    if otlp_endpoint:
        try:
            from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
                OTLPSpanExporter,
            )
            exporter = OTLPSpanExporter(endpoint=otlp_endpoint)
        except ImportError:
            exporter = ConsoleSpanExporter()
    else:
        exporter = ConsoleSpanExporter()

    provider.add_span_processor(BatchSpanProcessor(exporter))
    trace.set_tracer_provider(provider)
    _tracer_provider = provider

    return provider


def get_tracer(name: str = __name__) -> Tracer:
    """Get a named tracer instance."""
    return trace.get_tracer(name)


def get_current_trace_id() -> Optional[str]:
    """Get the current trace ID as hex string."""
    span = trace.get_current_span()
    span_context = span.get_span_context()
    
    if span_context.trace_id == 0:
        return None
    
    return format(span_context.trace_id, "032x")


def get_current_span_id() -> Optional[str]:
    """Get the current span ID as hex string."""
    span = trace.get_current_span()
    span_context = span.get_span_context()
    
    if span_context.span_id == 0:
        return None
    
    return format(span_context.span_id, "016x")


def inject_trace_context(headers: Optional[dict[str, str]] = None) -> dict[str, str]:
    """Inject trace context into headers for outbound requests."""
    carrier = headers if headers is not None else {}
    inject(carrier)
    return carrier


def extract_trace_context(headers: dict[str, Any]) -> Context:
    """Extract trace context from incoming headers."""
    return extract(headers)


def _headers_to_dict(headers: list[tuple[bytes, bytes]]) -> dict[str, str]:
    """Convert ASGI headers to dict for propagation."""
    return {
        key.decode("utf-8").lower(): value.decode("utf-8")
        for key, value in headers
    }


class TracingMiddleware:
    """
    ASGI middleware for OpenTelemetry tracing.
    
    Creates spans for HTTP requests and propagates trace context.
    """

    def __init__(
        self,
        app: Callable[..., Any],
        exclude_paths: Optional[list[str]] = None,
        tracer_name: str = "inference_service.http",
    ) -> None:
        self.app = app
        self.exclude_paths = exclude_paths or []
        self.tracer = get_tracer(tracer_name)

    async def __call__(
        self,
        scope: dict[str, Any],
        receive: Callable[..., Any],
        send: Callable[..., Any],
    ) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        method = scope.get("method", "GET")
        path = scope.get("path", "/")

        if path in self.exclude_paths:
            await self.app(scope, receive, send)
            return

        headers = scope.get("headers", [])
        headers_dict = _headers_to_dict(headers)
        parent_context = extract_trace_context(headers_dict)

        span_name = f"{method} {path}"
        status_code = 500

        async def send_wrapper(message: dict[str, Any]) -> None:
            nonlocal status_code
            if message["type"] == "http.response.start":
                status_code = message.get("status", 500)
            await send(message)

        with self.tracer.start_as_current_span(
            span_name,
            context=parent_context,
            kind=SpanKind.SERVER,
        ) as span:
            span.set_attribute("http.method", method)
            span.set_attribute("http.route", path)

            try:
                await self.app(scope, receive, send_wrapper)
                span.set_attribute("http.status_code", status_code)
            except Exception as e:
                span.set_attribute("http.status_code", 500)
                span.record_exception(e)
                raise
