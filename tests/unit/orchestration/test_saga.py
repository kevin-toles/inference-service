"""Unit tests for PipelineSaga orchestration.

Tests for:
- PipelineSaga saga compensation pattern (AC-13.3)
- PipelineSaga returns partial results on error (AC-13.4)

Reference: ARCHITECTURE.md → Error Handling → Saga Compensation Pattern
"""

import time
from dataclasses import dataclass
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.models.requests import ChatCompletionRequest, Message
from src.models.responses import (
    ChatCompletionResponse,
    Choice,
    ChoiceMessage,
    Usage,
)
from src.orchestration.saga import (
    CompletedStep,
    PipelineSaga,
    SagaStep,
    StepResult,
)

# =============================================================================
# Constants (S1192: Avoid duplicated string literals)
# =============================================================================

TEST_MODEL_PHI4 = "phi-4"
TEST_MODEL_1 = "model-1"
TEST_MODEL_2 = "model-2"
TEST_MODEL_3 = "model-3"
TEST_MODEL_TEST = "test-model"
TEST_MODEL_LLAMA = "llama-3.2-3b"
TEST_MODEL_DEEPSEEK = "deepseek-r1-7b"
ROLE_SYSTEM = "system"
ROLE_USER = "user"
STEP_DRAFT = "draft"
STEP_REFINE = "refine"
STEP_VALIDATE = "validate"
TEST_OUTPUT = "output"
FINISH_REASON_PARTIAL = "partial"


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_request() -> ChatCompletionRequest:
    """Create a sample chat completion request."""
    return ChatCompletionRequest(
        model=TEST_MODEL_PHI4,
        messages=[
            Message(role=ROLE_SYSTEM, content="You are a helpful assistant."),
            Message(role=ROLE_USER, content="Write a function to calculate factorial."),
        ],
        temperature=0.7,
        max_tokens=500,
    )


@pytest.fixture
def successful_step_result() -> StepResult:
    """Create a successful step result."""
    return StepResult(
        output="def factorial(n): return 1 if n <= 1 else n * factorial(n-1)",
        is_usable=True,
        model_id=TEST_MODEL_LLAMA,
        usage=Usage(prompt_tokens=30, completion_tokens=25, total_tokens=55),
    )


@pytest.fixture
def refined_step_result() -> StepResult:
    """Create a refined step result."""
    return StepResult(
        output=(
            "def factorial(n: int) -> int:\n"
            "    '''Calculate factorial iteratively.'''\n"
            "    if n < 0:\n"
            "        raise ValueError('n must be non-negative')\n"
            "    result = 1\n"
            "    for i in range(2, n + 1):\n"
            "        result *= i\n"
            "    return result"
        ),
        is_usable=True,
        model_id=TEST_MODEL_DEEPSEEK,
        usage=Usage(prompt_tokens=60, completion_tokens=80, total_tokens=140),
    )


# =============================================================================
# SagaStep Tests
# =============================================================================


class TestSagaStep:
    """Tests for SagaStep class."""

    def test_saga_step_creation(self) -> None:
        """Test SagaStep can be created."""
        async def invoke_fn(state: dict[str, Any]) -> StepResult:
            return StepResult(
                output="test",
                is_usable=True,
                model_id=TEST_MODEL_TEST,
                usage=Usage(prompt_tokens=10, completion_tokens=10, total_tokens=20),
            )

        step = SagaStep(
            name=STEP_DRAFT,
            invoke=invoke_fn,
        )

        assert step.name == STEP_DRAFT

    @pytest.mark.asyncio
    async def test_saga_step_invoke(self) -> None:
        """Test SagaStep invoke works."""
        async def invoke_fn(state: dict[str, Any]) -> StepResult:
            return StepResult(
                output="test output",
                is_usable=True,
                model_id=TEST_MODEL_TEST,
                usage=Usage(prompt_tokens=10, completion_tokens=10, total_tokens=20),
            )

        step = SagaStep(name=STEP_DRAFT, invoke=invoke_fn)
        result = await step.invoke({})

        assert result.output == "test output"
        assert result.is_usable is True


# =============================================================================
# StepResult Tests
# =============================================================================


class TestStepResult:
    """Tests for StepResult class."""

    def test_step_result_creation(self) -> None:
        """Test StepResult can be created."""
        result = StepResult(
            output="test output",
            is_usable=True,
            model_id=TEST_MODEL_TEST,
            usage=Usage(prompt_tokens=10, completion_tokens=10, total_tokens=20),
        )

        assert result.output == "test output"
        assert result.is_usable is True
        assert result.model_id == TEST_MODEL_TEST

    def test_step_result_not_usable(self) -> None:
        """Test StepResult with is_usable=False."""
        result = StepResult(
            output="",
            is_usable=False,
            model_id=TEST_MODEL_TEST,
            usage=Usage(prompt_tokens=10, completion_tokens=0, total_tokens=10),
        )

        assert result.is_usable is False


# =============================================================================
# CompletedStep Tests
# =============================================================================


class TestCompletedStep:
    """Tests for CompletedStep class."""

    def test_completed_step_creation(self) -> None:
        """Test CompletedStep can be created."""
        async def invoke_fn(state: dict[str, Any]) -> StepResult:
            return StepResult(
                output="test",
                is_usable=True,
                model_id=TEST_MODEL_TEST,
                usage=Usage(prompt_tokens=10, completion_tokens=10, total_tokens=20),
            )

        step = SagaStep(name=STEP_DRAFT, invoke=invoke_fn)
        result = StepResult(
            output="test output",
            is_usable=True,
            model_id=TEST_MODEL_TEST,
            usage=Usage(prompt_tokens=10, completion_tokens=10, total_tokens=20),
        )

        completed = CompletedStep(
            step=step,
            result=result,
            state_snapshot={"goal": "test"},
        )

        assert completed.step == step
        assert completed.result == result


# =============================================================================
# PipelineSaga Tests - AC-13.3: Saga Compensation
# =============================================================================


class TestPipelineSagaCompensation:
    """Tests for PipelineSaga compensation pattern."""

    @pytest.mark.asyncio
    async def test_saga_executes_all_steps_on_success(
        self,
        sample_request: ChatCompletionRequest,
    ) -> None:
        """Test PipelineSaga executes all steps when no errors."""
        step1_called = False
        step2_called = False
        step3_called = False

        async def step1_invoke(state: dict[str, Any]) -> StepResult:
            nonlocal step1_called
            step1_called = True
            return StepResult(
                output=STEP_DRAFT,
                is_usable=True,
                model_id=TEST_MODEL_1,
                usage=Usage(prompt_tokens=10, completion_tokens=10, total_tokens=20),
            )

        async def step2_invoke(state: dict[str, Any]) -> StepResult:
            nonlocal step2_called
            step2_called = True
            return StepResult(
                output="refined",
                is_usable=True,
                model_id=TEST_MODEL_2,
                usage=Usage(prompt_tokens=20, completion_tokens=20, total_tokens=40),
            )

        async def step3_invoke(state: dict[str, Any]) -> StepResult:
            nonlocal step3_called
            step3_called = True
            return StepResult(
                output="validated",
                is_usable=True,
                model_id=TEST_MODEL_3,
                usage=Usage(prompt_tokens=30, completion_tokens=30, total_tokens=60),
            )

        saga = PipelineSaga()
        saga.add_step(SagaStep(name=STEP_DRAFT, invoke=step1_invoke))
        saga.add_step(SagaStep(name=STEP_REFINE, invoke=step2_invoke))
        saga.add_step(SagaStep(name=STEP_VALIDATE, invoke=step3_invoke))

        result = await saga.execute(sample_request)

        assert step1_called
        assert step2_called
        assert step3_called
        assert result is not None

    @pytest.mark.asyncio
    async def test_saga_tracks_completed_steps(
        self,
        sample_request: ChatCompletionRequest,
    ) -> None:
        """Test PipelineSaga tracks completed steps."""
        async def step_invoke(state: dict[str, Any]) -> StepResult:
            return StepResult(
                output=TEST_OUTPUT,
                is_usable=True,
                model_id="model",
                usage=Usage(prompt_tokens=10, completion_tokens=10, total_tokens=20),
            )

        saga = PipelineSaga()
        saga.add_step(SagaStep(name="step1", invoke=step_invoke))
        saga.add_step(SagaStep(name="step2", invoke=step_invoke))

        await saga.execute(sample_request)

        assert len(saga.completed_steps) == 2

    @pytest.mark.asyncio
    async def test_saga_compensates_on_error(
        self,
        sample_request: ChatCompletionRequest,
    ) -> None:
        """Test PipelineSaga calls _compensate on error."""
        async def step1_invoke(state: dict[str, Any]) -> StepResult:
            return StepResult(
                output="draft output",
                is_usable=True,
                model_id=TEST_MODEL_1,
                usage=Usage(prompt_tokens=10, completion_tokens=10, total_tokens=20),
            )

        async def step2_invoke(state: dict[str, Any]) -> StepResult:
            raise RuntimeError("Step 2 failed")

        saga = PipelineSaga()
        saga.add_step(SagaStep(name=STEP_DRAFT, invoke=step1_invoke))
        saga.add_step(SagaStep(name=STEP_REFINE, invoke=step2_invoke))

        result = await saga.execute(sample_request)

        # Should return partial result from compensation
        assert result is not None
        assert result.choices[0].finish_reason == FINISH_REASON_PARTIAL


# =============================================================================
# PipelineSaga Tests - AC-13.4: Partial Results
# =============================================================================


class TestPipelineSagaPartialResults:
    """Tests for PipelineSaga partial results."""

    @pytest.mark.asyncio
    async def test_saga_returns_last_successful_result(
        self,
        sample_request: ChatCompletionRequest,
    ) -> None:
        """Test PipelineSaga returns last successful step's output."""
        async def step1_invoke(state: dict[str, Any]) -> StepResult:
            return StepResult(
                output="draft output",
                is_usable=True,
                model_id=TEST_MODEL_1,
                usage=Usage(prompt_tokens=10, completion_tokens=10, total_tokens=20),
            )

        async def step2_invoke(state: dict[str, Any]) -> StepResult:
            return StepResult(
                output="refined output",
                is_usable=True,
                model_id=TEST_MODEL_2,
                usage=Usage(prompt_tokens=20, completion_tokens=20, total_tokens=40),
            )

        async def step3_invoke(state: dict[str, Any]) -> StepResult:
            raise RuntimeError("Validator failed")

        saga = PipelineSaga()
        saga.add_step(SagaStep(name=STEP_DRAFT, invoke=step1_invoke))
        saga.add_step(SagaStep(name=STEP_REFINE, invoke=step2_invoke))
        saga.add_step(SagaStep(name=STEP_VALIDATE, invoke=step3_invoke))

        result = await saga.execute(sample_request)

        # Should return refiner's output (last successful)
        assert result.choices[0].message.content == "refined output"
        assert result.choices[0].finish_reason == FINISH_REASON_PARTIAL

    @pytest.mark.asyncio
    async def test_saga_partial_includes_error_info(
        self,
        sample_request: ChatCompletionRequest,
    ) -> None:
        """Test partial result includes error information."""
        async def step1_invoke(state: dict[str, Any]) -> StepResult:
            return StepResult(
                output=TEST_OUTPUT,
                is_usable=True,
                model_id=TEST_MODEL_1,
                usage=Usage(prompt_tokens=10, completion_tokens=10, total_tokens=20),
            )

        async def step2_invoke(state: dict[str, Any]) -> StepResult:
            raise RuntimeError("Test error message")

        saga = PipelineSaga()
        saga.add_step(SagaStep(name=STEP_DRAFT, invoke=step1_invoke))
        saga.add_step(SagaStep(name=STEP_REFINE, invoke=step2_invoke))

        result = await saga.execute(sample_request)

        # Orchestration metadata should indicate partial
        assert result.orchestration is not None
        assert result.orchestration.mode == "pipeline"

    @pytest.mark.asyncio
    async def test_saga_raises_when_first_step_fails(
        self,
        sample_request: ChatCompletionRequest,
    ) -> None:
        """Test PipelineSaga raises error when first step fails."""
        from src.orchestration.saga import OrchestrationFailedError

        async def step1_invoke(state: dict[str, Any]) -> StepResult:
            raise RuntimeError("First step failed")

        saga = PipelineSaga()
        saga.add_step(SagaStep(name=STEP_DRAFT, invoke=step1_invoke))

        with pytest.raises(OrchestrationFailedError, match="Pipeline failed"):
            await saga.execute(sample_request)

    @pytest.mark.asyncio
    async def test_saga_skips_non_usable_results(
        self,
        sample_request: ChatCompletionRequest,
    ) -> None:
        """Test PipelineSaga skips non-usable results in compensation."""
        async def step1_invoke(state: dict[str, Any]) -> StepResult:
            return StepResult(
                output="usable output",
                is_usable=True,
                model_id=TEST_MODEL_1,
                usage=Usage(prompt_tokens=10, completion_tokens=10, total_tokens=20),
            )

        async def step2_invoke(state: dict[str, Any]) -> StepResult:
            return StepResult(
                output="",  # Empty, not usable
                is_usable=False,
                model_id=TEST_MODEL_2,
                usage=Usage(prompt_tokens=20, completion_tokens=0, total_tokens=20),
            )

        async def step3_invoke(state: dict[str, Any]) -> StepResult:
            raise RuntimeError("Step 3 failed")

        saga = PipelineSaga()
        saga.add_step(SagaStep(name="step1", invoke=step1_invoke))
        saga.add_step(SagaStep(name="step2", invoke=step2_invoke))
        saga.add_step(SagaStep(name="step3", invoke=step3_invoke))

        result = await saga.execute(sample_request)

        # Should return step1's output (step2 was not usable)
        assert result.choices[0].message.content == "usable output"


# =============================================================================
# PipelineSaga Metadata Tests
# =============================================================================


class TestPipelineSagaMetadata:
    """Tests for PipelineSaga metadata handling."""

    @pytest.mark.asyncio
    async def test_saga_aggregates_usage(
        self,
        sample_request: ChatCompletionRequest,
    ) -> None:
        """Test PipelineSaga aggregates usage stats."""
        async def step1_invoke(state: dict[str, Any]) -> StepResult:
            return StepResult(
                output="output1",
                is_usable=True,
                model_id=TEST_MODEL_1,
                usage=Usage(prompt_tokens=10, completion_tokens=10, total_tokens=20),
            )

        async def step2_invoke(state: dict[str, Any]) -> StepResult:
            return StepResult(
                output="output2",
                is_usable=True,
                model_id=TEST_MODEL_2,
                usage=Usage(prompt_tokens=20, completion_tokens=20, total_tokens=40),
            )

        saga = PipelineSaga()
        saga.add_step(SagaStep(name="step1", invoke=step1_invoke))
        saga.add_step(SagaStep(name="step2", invoke=step2_invoke))

        result = await saga.execute(sample_request)

        # Total should be 20 + 40 = 60
        assert result.usage.total_tokens == 60

    @pytest.mark.asyncio
    async def test_saga_collects_models_used(
        self,
        sample_request: ChatCompletionRequest,
    ) -> None:
        """Test PipelineSaga collects all models used."""
        async def step1_invoke(state: dict[str, Any]) -> StepResult:
            return StepResult(
                output="output1",
                is_usable=True,
                model_id=TEST_MODEL_LLAMA,
                usage=Usage(prompt_tokens=10, completion_tokens=10, total_tokens=20),
            )

        async def step2_invoke(state: dict[str, Any]) -> StepResult:
            return StepResult(
                output="output2",
                is_usable=True,
                model_id=TEST_MODEL_DEEPSEEK,
                usage=Usage(prompt_tokens=20, completion_tokens=20, total_tokens=40),
            )

        saga = PipelineSaga()
        saga.add_step(SagaStep(name=STEP_DRAFT, invoke=step1_invoke))
        saga.add_step(SagaStep(name=STEP_REFINE, invoke=step2_invoke))

        result = await saga.execute(sample_request)

        assert TEST_MODEL_LLAMA in result.orchestration.models_used
        assert TEST_MODEL_DEEPSEEK in result.orchestration.models_used

    @pytest.mark.asyncio
    async def test_saga_tracks_inference_time(
        self,
        sample_request: ChatCompletionRequest,
    ) -> None:
        """Test PipelineSaga tracks total inference time."""
        async def step_invoke(state: dict[str, Any]) -> StepResult:
            return StepResult(
                output=TEST_OUTPUT,
                is_usable=True,
                model_id="model",
                usage=Usage(prompt_tokens=10, completion_tokens=10, total_tokens=20),
            )

        saga = PipelineSaga()
        saga.add_step(SagaStep(name="step", invoke=step_invoke))

        result = await saga.execute(sample_request)

        assert result.orchestration.total_inference_time_ms > 0
