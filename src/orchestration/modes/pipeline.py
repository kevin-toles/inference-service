"""Pipeline mode orchestration - Draft → Refine → Validate flow.

PipelineMode implements sequential multi-model orchestration where:
1. Drafter (Fast model) creates quick initial draft
2. Refiner (Specialist model) improves and refines the draft
3. Validator (Primary model) validates and finalizes the response

Flow:
    Request → Fast(draft) → Specialist(refine) → Primary(validate) → Response

Per ARCHITECTURE.md Orchestration Modes (pipeline):
- pipeline: Sequential stages
- Min Models: 2-3
- Flow: Request → Fast(draft) → Specialist(refine) → Primary(validate) → Response

Reference: WBS-INF13 AC-13.1, AC-13.2, AC-13.3, AC-13.4
"""

import time
from typing import Any

from src.models.requests import ChatCompletionRequest, Message
from src.models.responses import (
    ChatCompletionResponse,
    OrchestrationMetadata,
)
from src.orchestration.saga import (
    OrchestrationFailedError,
    PipelineSaga,
    SagaStep,
    StepResult,
)
from src.providers.base import InferenceProvider


# =============================================================================
# Constants
# =============================================================================


# Token limit for phi-4 context - content larger than this needs compression
PHI4_CONTEXT_LIMIT = 16384

DRAFT_SYSTEM_PROMPT = """Generate a quick initial response to the user's request.
Focus on getting the main points correct. This is a draft that will be refined."""

REFINE_SYSTEM_PROMPT = """Refine and improve the following draft response.
Add type hints if code, improve clarity, handle edge cases, and ensure correctness.

Draft to refine:
{draft}

Provide an improved version."""

VALIDATE_SYSTEM_PROMPT = """Review and validate the following refined response.
Ensure it is correct, complete, and well-formatted. Make any final improvements.

Response to validate:
{refined}

Provide the final validated response."""


# =============================================================================
# PipelineMode Implementation
# =============================================================================


class PipelineMode:
    """Pipeline mode orchestration - Draft → Refine → Validate.

    Coordinates 2-3 models in a sequential pipeline:
    1. Drafter creates quick initial response
    2. Refiner (optional) improves and adds detail
    3. Validator finalizes and validates

    Uses PipelineSaga for compensation on failures.

    AC-13.1: Draft → Refine → Validate flow
    AC-13.2: Compresses between steps if needed
    AC-13.3: Saga compensation on failure
    AC-13.4: Returns partial result on error

    Attributes:
        drafter: Provider for drafting (Fast model).
        refiner: Optional provider for refining (Specialist model).
        validator: Provider for validation (Primary model).

    Example:
        drafter = LlamaCppProvider(model_path="llama-3.2-3b.gguf")
        refiner = LlamaCppProvider(model_path="deepseek-r1-7b.gguf")
        validator = LlamaCppProvider(model_path="phi-4.gguf")
        mode = PipelineMode(drafter=drafter, refiner=refiner, validator=validator)
        response = await mode.execute(request)
    """

    def __init__(
        self,
        drafter: InferenceProvider,
        validator: InferenceProvider,
        refiner: InferenceProvider | None = None,
    ) -> None:
        """Initialize PipelineMode with providers.

        Args:
            drafter: Provider for drafting (Fast model).
            validator: Provider for validation (Primary model).
            refiner: Optional provider for refining (Specialist model).
        """
        self._drafter = drafter
        self._refiner = refiner
        self._validator = validator

    @property
    def drafter(self) -> InferenceProvider:
        """Get the drafter provider."""
        return self._drafter

    @property
    def refiner(self) -> InferenceProvider | None:
        """Get the refiner provider."""
        return self._refiner

    @property
    def validator(self) -> InferenceProvider:
        """Get the validator provider."""
        return self._validator

    @property
    def drafter_model_id(self) -> str:
        """Get the drafter model ID."""
        return self._drafter.model_info.model_id

    @property
    def refiner_model_id(self) -> str | None:
        """Get the refiner model ID."""
        if self._refiner is None:
            return None
        return self._refiner.model_info.model_id

    @property
    def validator_model_id(self) -> str:
        """Get the validator model ID."""
        return self._validator.model_info.model_id

    async def execute(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        """Execute pipeline mode orchestration.

        Flow:
        1. Draft: Fast model generates initial response
        2. Refine (optional): Specialist model improves the draft
        3. Validate: Primary model validates and finalizes

        Uses PipelineSaga for automatic compensation on failure.

        AC-13.1: Implements Draft → Refine → Validate flow.
        AC-13.2: Compresses content if needed between steps.
        AC-13.3: Saga compensation on failure.
        AC-13.4: Returns partial result on error.

        Args:
            request: Chat completion request.

        Returns:
            ChatCompletionResponse with orchestration metadata.

        Raises:
            RuntimeError: If drafter fails (no partial available).
        """
        start_time = time.perf_counter()

        # Build saga with steps
        saga = PipelineSaga()

        # Step 1: Draft
        saga.add_step(
            SagaStep(
                name="draft",
                invoke=self._create_draft_step(request),
            )
        )

        # Step 2: Refine (optional)
        if self._refiner is not None:
            saga.add_step(
                SagaStep(
                    name="refine",
                    invoke=self._create_refine_step(request),
                )
            )

        # Step 3: Validate
        saga.add_step(
            SagaStep(
                name="validate",
                invoke=self._create_validate_step(request),
            )
        )

        # Execute saga (handles compensation automatically)
        try:
            result = await saga.execute(request)
        except OrchestrationFailedError:
            # Re-raise as RuntimeError for first step failure
            raise RuntimeError("Drafter failed") from None

        # Update orchestration metadata with all models
        models_used = self._collect_models_used()
        inference_time_ms = (time.perf_counter() - start_time) * 1000

        result.orchestration = OrchestrationMetadata(
            mode="pipeline",
            models_used=models_used,
            total_inference_time_ms=inference_time_ms,
            rounds=None,
            final_score=None,
            agreement_score=None,
        )

        return result

    def _create_draft_step(
        self, request: ChatCompletionRequest
    ) -> Any:  # Returns Callable
        """Create the draft step invoke function.

        Args:
            request: Original request.

        Returns:
            Async callable for saga step.
        """

        async def invoke(_state: dict[str, Any]) -> StepResult:
            # Build draft request
            draft_request = ChatCompletionRequest(
                model=self.drafter_model_id,
                messages=[
                    Message(role="system", content=DRAFT_SYSTEM_PROMPT),
                    *request.messages,
                ],
                temperature=request.temperature,
                max_tokens=request.max_tokens,
            )

            response = await self._drafter.generate(draft_request)
            content = self._extract_content(response)

            return StepResult(
                output=content,
                is_usable=True,
                model_id=self.drafter_model_id,
                usage=response.usage,
            )

        return invoke

    def _create_refine_step(
        self, request: ChatCompletionRequest
    ) -> Any:  # Returns Callable
        """Create the refine step invoke function.

        Args:
            request: Original request.

        Returns:
            Async callable for saga step.
        """

        async def invoke(state: dict[str, Any]) -> StepResult:
            draft_content = state.get("current_output", "")

            # Compress if needed (AC-13.2)
            compressed_content = self._maybe_compress(draft_content)

            # Build refine request
            refine_prompt = REFINE_SYSTEM_PROMPT.format(draft=compressed_content)
            refine_request = ChatCompletionRequest(
                model=self.refiner_model_id or "",
                messages=[
                    Message(role="system", content=refine_prompt),
                    *request.messages,
                ],
                temperature=request.temperature,
                max_tokens=request.max_tokens,
            )

            if self._refiner is None:
                raise RuntimeError("Refiner not configured")

            response = await self._refiner.generate(refine_request)
            content = self._extract_content(response)

            return StepResult(
                output=content,
                is_usable=True,
                model_id=self.refiner_model_id or "",
                usage=response.usage,
            )

        return invoke

    def _create_validate_step(
        self, request: ChatCompletionRequest
    ) -> Any:  # Returns Callable
        """Create the validate step invoke function.

        Args:
            request: Original request.

        Returns:
            Async callable for saga step.
        """

        async def invoke(state: dict[str, Any]) -> StepResult:
            refined_content = state.get("current_output", "")

            # Compress if needed for phi-4's context limit (AC-13.2)
            compressed_content = self._maybe_compress(
                refined_content, PHI4_CONTEXT_LIMIT
            )

            # Build validate request
            validate_prompt = VALIDATE_SYSTEM_PROMPT.format(refined=compressed_content)
            validate_request = ChatCompletionRequest(
                model=self.validator_model_id,
                messages=[
                    Message(role="system", content=validate_prompt),
                    *request.messages,
                ],
                temperature=request.temperature,
                max_tokens=request.max_tokens,
            )

            response = await self._validator.generate(validate_request)
            content = self._extract_content(response)

            return StepResult(
                output=content,
                is_usable=True,
                model_id=self.validator_model_id,
                usage=response.usage,
            )

        return invoke

    def _extract_content(self, response: ChatCompletionResponse) -> str:
        """Extract content from response.

        Args:
            response: Chat completion response.

        Returns:
            Content string or empty string if None.
        """
        if response.choices and response.choices[0].message.content:
            return response.choices[0].message.content
        return ""

    def _maybe_compress(
        self, content: str, limit: int = PHI4_CONTEXT_LIMIT
    ) -> str:
        """Compress content if it exceeds limit.

        Simple truncation-based compression for now.
        In production, would use fit_to_budget() from context.py.

        AC-13.2: Compresses between steps if needed.

        Args:
            content: Content to potentially compress.
            limit: Token limit (approximate character-based).

        Returns:
            Original or compressed content.
        """
        # Simple character-based approximation (4 chars ≈ 1 token)
        char_limit = limit * 4

        if len(content) <= char_limit:
            return content

        # Truncate with ellipsis
        return content[: char_limit - 3] + "..."

    def _collect_models_used(self) -> list[str]:
        """Collect all model IDs used in pipeline.

        Returns:
            List of model IDs.
        """
        models = [self.drafter_model_id]
        if self._refiner is not None:
            models.append(self.refiner_model_id or "")
        models.append(self.validator_model_id)
        return models
