"""Saga orchestration for multi-model pipelines with compensation.

Implements the Saga Compensation Pattern for pipeline orchestration,
enabling graceful degradation and partial result recovery on failures.

Per ARCHITECTURE.md Error Handling → Saga Compensation Pattern:
- Tracks completed steps for rollback
- Returns partial results when possible
- Aggregates usage stats across steps

Reference: WBS-INF13 AC-13.3, AC-13.4
"""

import time
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from src.models.requests import ChatCompletionRequest
from src.models.responses import (
    ChatCompletionResponse,
    Choice,
    ChoiceMessage,
    OrchestrationMetadata,
    Usage,
)


# =============================================================================
# Exceptions
# =============================================================================


class OrchestrationFailedError(Exception):
    """Raised when orchestration fails without usable partial result."""

    pass


# =============================================================================
# Data Structures
# =============================================================================


@dataclass
class StepResult:
    """Result from a saga step execution.

    Attributes:
        output: Generated content from this step.
        is_usable: Whether this output can be used as partial result.
        model_id: ID of the model that produced this output.
        usage: Token usage statistics for this step.
    """

    output: str
    is_usable: bool
    model_id: str
    usage: Usage


@dataclass
class SagaStep:
    """A step in the saga pipeline.

    Attributes:
        name: Human-readable step name (e.g., "draft", "refine", "validate").
        invoke: Async function that executes this step.
    """

    name: str
    invoke: Callable[[dict[str, Any]], Awaitable[StepResult]]


@dataclass
class CompletedStep:
    """Record of a completed step with its result and state.

    Attributes:
        step: The step that was executed.
        result: The result from executing the step.
        state_snapshot: Copy of state after step completion.
    """

    step: SagaStep
    result: StepResult
    state_snapshot: dict[str, Any]


# =============================================================================
# PipelineSaga Implementation
# =============================================================================


class PipelineSaga:
    """Saga orchestration for multi-model pipelines with compensation.

    Implements saga pattern for pipeline execution:
    1. Execute steps sequentially
    2. Track completed steps for potential rollback
    3. On failure, compensate by returning best partial result

    AC-13.3: Saga compensation on failure
    AC-13.4: Returns partial result with finish_reason="partial"

    Example:
        saga = PipelineSaga()
        saga.add_step(SagaStep(name="draft", invoke=draft_fn))
        saga.add_step(SagaStep(name="refine", invoke=refine_fn))
        saga.add_step(SagaStep(name="validate", invoke=validate_fn))
        response = await saga.execute(request)
    """

    def __init__(self) -> None:
        """Initialize PipelineSaga with empty steps list."""
        self.steps: list[SagaStep] = []
        self.completed_steps: list[CompletedStep] = []

    def add_step(self, step: SagaStep) -> None:
        """Add a step to the pipeline.

        Args:
            step: SagaStep to add to the pipeline.
        """
        self.steps.append(step)

    async def execute(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        """Execute pipeline with compensation on failure.

        Executes steps sequentially, tracking completed steps.
        On failure, calls _compensate to return best partial result.

        Args:
            request: Chat completion request.

        Returns:
            ChatCompletionResponse with full or partial result.

        Raises:
            OrchestrationFailedError: If pipeline fails with no usable partial.
        """
        start_time = time.perf_counter()
        self.completed_steps = []  # Reset for new execution

        state: dict[str, Any] = {
            "request": request,
            "goal": self._extract_goal(request),
            "current_output": "",
        }

        for step in self.steps:
            try:
                result = await step.invoke(state)
                state["current_output"] = result.output
                self.completed_steps.append(
                    CompletedStep(
                        step=step,
                        result=result,
                        state_snapshot=state.copy(),
                    )
                )
            except Exception as e:
                # Compensate and return partial result
                return await self._compensate(state, e, start_time)

        # All steps completed successfully
        return self._build_response(state, start_time)

    async def _compensate(
        self,
        _state: dict[str, Any],
        error: Exception,
        start_time: float,
    ) -> ChatCompletionResponse:
        """Roll back and return best partial result.

        Finds the last successful step with usable output and returns
        that as a partial result.

        Args:
            state: Current pipeline state.
            error: The exception that caused failure.
            start_time: Start time for timing calculation.

        Returns:
            ChatCompletionResponse with partial result.

        Raises:
            OrchestrationFailedError: If no usable partial result exists.
        """
        # Find last successful step with usable output
        for completed in reversed(self.completed_steps):
            if completed.result.is_usable:
                inference_time_ms = (time.perf_counter() - start_time) * 1000
                return ChatCompletionResponse(
                    id=f"chatcmpl-{uuid.uuid4().hex[:12]}",
                    created=int(time.time()),
                    model=completed.result.model_id,
                    choices=[
                        Choice(
                            index=0,
                            message=ChoiceMessage(
                                role="assistant",
                                content=completed.result.output,
                            ),
                            finish_reason="partial",
                        )
                    ],
                    usage=self._aggregate_usage(),
                    orchestration=OrchestrationMetadata(
                        mode="pipeline",
                        models_used=self._collect_models_used(),
                        total_inference_time_ms=inference_time_ms,
                        rounds=None,
                        final_score=None,
                        agreement_score=None,
                    ),
                )

        # No usable partial result
        raise OrchestrationFailedError(f"Pipeline failed at step 1: {error}")

    def _build_response(
        self,
        _state: dict[str, Any],
        start_time: float,
    ) -> ChatCompletionResponse:
        """Build final response from completed pipeline.

        Args:
            state: Final pipeline state.
            start_time: Start time for timing calculation.

        Returns:
            ChatCompletionResponse with complete result.
        """
        inference_time_ms = (time.perf_counter() - start_time) * 1000
        last_completed = self.completed_steps[-1]

        return ChatCompletionResponse(
            id=f"chatcmpl-{uuid.uuid4().hex[:12]}",
            created=int(time.time()),
            model=last_completed.result.model_id,
            choices=[
                Choice(
                    index=0,
                    message=ChoiceMessage(
                        role="assistant",
                        content=last_completed.result.output,
                    ),
                    finish_reason="stop",
                )
            ],
            usage=self._aggregate_usage(),
            orchestration=OrchestrationMetadata(
                mode="pipeline",
                models_used=self._collect_models_used(),
                total_inference_time_ms=inference_time_ms,
                rounds=None,
                final_score=None,
                agreement_score=None,
            ),
        )

    def _aggregate_usage(self) -> Usage:
        """Aggregate usage stats from all completed steps.

        Returns:
            Combined Usage object.
        """
        total_prompt = 0
        total_completion = 0

        for completed in self.completed_steps:
            total_prompt += completed.result.usage.prompt_tokens
            total_completion += completed.result.usage.completion_tokens

        return Usage(
            prompt_tokens=total_prompt,
            completion_tokens=total_completion,
            total_tokens=total_prompt + total_completion,
        )

    def _collect_models_used(self) -> list[str]:
        """Collect unique model IDs from completed steps.

        Returns:
            List of model IDs used.
        """
        models: list[str] = []
        for completed in self.completed_steps:
            if completed.result.model_id not in models:
                models.append(completed.result.model_id)
        return models

    def _extract_goal(self, request: ChatCompletionRequest) -> str:
        """Extract the user's goal from request.

        Args:
            request: Chat completion request.

        Returns:
            User message content.
        """
        for message in reversed(request.messages):
            if message.role == "user" and message.content:
                return message.content
        return ""
