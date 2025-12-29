"""Debate mode orchestration - Parallel generation → Reconcile flow.

DebateMode implements multi-model orchestration where:
1. Two participants generate responses in parallel
2. Outputs are compared for agreement percentage
3. Reconciler synthesizes final answer from both outputs

Flow:
    Request → [A,B](parallel) → Compare → Reconcile → Response

Per ARCHITECTURE.md Orchestration Modes (debate):
- debate: Parallel then reconcile
- Min Models: 2
- Flow: Request → [A,B](parallel) → Compare → Reconcile → Response

Reference: WBS-INF14 AC-14.1, AC-14.2, AC-14.3, AC-14.4
"""

import asyncio
import time
import uuid

from src.models.requests import ChatCompletionRequest, Message
from src.models.responses import (
    ChatCompletionResponse,
    OrchestrationMetadata,
    Usage,
)
from src.providers.base import InferenceProvider


# =============================================================================
# Constants
# =============================================================================


RECONCILE_SYSTEM_PROMPT = """You are synthesizing a final answer from two model responses.

Both responses attempt to answer the same question. Your task is to:
1. Identify points of agreement between the responses
2. Resolve any conflicts by choosing the most accurate/complete information
3. Create a coherent, well-structured final answer that combines the best elements

Do NOT mention that you are synthesizing from multiple responses.
Simply provide the best possible answer."""

RECONCILE_USER_PROMPT_TEMPLATE = """Original question: {question}

Response A:
{response_a}

Response B:
{response_b}

Please synthesize a final answer that combines the best information from both responses."""


# =============================================================================
# Agreement Calculation
# =============================================================================


def calculate_agreement(text_a: str, text_b: str) -> float:
    """Calculate agreement percentage between two texts using word overlap.

    Uses Jaccard similarity on word sets (case-insensitive).

    Args:
        text_a: First text to compare.
        text_b: Second text to compare.

    Returns:
        Agreement score between 0.0 and 1.0.

    Example:
        >>> calculate_agreement("Paris is the capital", "Paris is capital")
        0.75  # 3 words in common / 4 total unique words
    """
    # Handle empty strings
    if not text_a and not text_b:
        return 1.0  # Vacuous truth - empty equals empty
    if not text_a or not text_b:
        return 0.0  # One empty, one not

    # Normalize and tokenize
    words_a = set(text_a.lower().split())
    words_b = set(text_b.lower().split())

    # Jaccard similarity: intersection / union
    intersection = words_a & words_b
    union = words_a | words_b

    if not union:
        return 1.0

    return len(intersection) / len(union)


# =============================================================================
# DebateMode Implementation
# =============================================================================


class DebateMode:
    """Debate mode orchestration - Parallel generation → Reconcile.

    Coordinates multiple models in a parallel-then-reconcile pattern:
    1. Both participants generate responses in parallel using asyncio.gather
    2. Outputs are compared to calculate agreement percentage
    3. Reconciler synthesizes final answer from both outputs

    AC-14.1: Parallel generation → Reconcile flow
    AC-14.2: Uses asyncio.gather for parallel execution
    AC-14.3: Compares outputs for agreement percentage
    AC-14.4: Reconciler synthesizes final answer

    Attributes:
        participant_a: First participant provider (Model A).
        participant_b: Second participant provider (Model B).
        reconciler: Provider for reconciliation (often same as participant_a).

    Example:
        model_a = LlamaCppProvider(model_path="phi-4.gguf")
        model_b = LlamaCppProvider(model_path="deepseek-r1-7b.gguf")
        mode = DebateMode(
            participant_a=model_a,
            participant_b=model_b,
            reconciler=model_a,  # phi-4 reconciles per ARCHITECTURE.md
        )
        response = await mode.execute(request)
    """

    def __init__(
        self,
        participant_a: InferenceProvider,
        participant_b: InferenceProvider,
        reconciler: InferenceProvider,
    ) -> None:
        """Initialize DebateMode with participant and reconciler providers.

        Args:
            participant_a: First participant provider (Model A).
            participant_b: Second participant provider (Model B).
            reconciler: Provider for reconciliation. Can be same as a participant.
        """
        self._participant_a = participant_a
        self._participant_b = participant_b
        self._reconciler = reconciler

    # -------------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------------

    @property
    def participant_a(self) -> InferenceProvider:
        """Get participant A provider."""
        return self._participant_a

    @property
    def participant_b(self) -> InferenceProvider:
        """Get participant B provider."""
        return self._participant_b

    @property
    def reconciler(self) -> InferenceProvider:
        """Get reconciler provider."""
        return self._reconciler

    @property
    def participant_a_model_id(self) -> str:
        """Get model ID for participant A."""
        return self._participant_a.model_info.model_id

    @property
    def participant_b_model_id(self) -> str:
        """Get model ID for participant B."""
        return self._participant_b.model_info.model_id

    @property
    def reconciler_model_id(self) -> str:
        """Get model ID for reconciler."""
        return self._reconciler.model_info.model_id

    # -------------------------------------------------------------------------
    # Main Execution
    # -------------------------------------------------------------------------

    async def execute(
        self, request: ChatCompletionRequest
    ) -> ChatCompletionResponse:
        """Execute debate mode orchestration.

        Steps:
        1. Generate responses from both participants in parallel
        2. Compare outputs for agreement percentage
        3. Call reconciler to synthesize final answer
        4. Build response with aggregated metadata

        Args:
            request: Original chat completion request.

        Returns:
            ChatCompletionResponse with reconciled content and metadata.

        Raises:
            Exception: If either participant fails during generation.
        """
        start_time = time.perf_counter()

        # Phase 1: Parallel generation using asyncio.gather (AC-14.2)
        response_a, response_b = await self._parallel_generate(request)

        # Extract content from responses
        content_a = self._extract_content(response_a)
        content_b = self._extract_content(response_b)

        # Phase 2: Compare outputs for agreement (AC-14.3)
        agreement = calculate_agreement(content_a, content_b)

        # Phase 3: Reconcile (AC-14.4)
        reconciled_response = await self._reconcile(
            request, content_a, content_b
        )

        # Build final response with metadata
        return self._build_response(
            reconciled_response=reconciled_response,
            response_a=response_a,
            response_b=response_b,
            agreement=agreement,
            start_time=start_time,
        )

    # -------------------------------------------------------------------------
    # Phase 1: Parallel Generation
    # -------------------------------------------------------------------------

    async def _parallel_generate(
        self, request: ChatCompletionRequest
    ) -> tuple[ChatCompletionResponse, ChatCompletionResponse]:
        """Generate responses from both participants in parallel.

        Uses asyncio.gather to run both generations concurrently.

        Args:
            request: Original request to send to both participants.

        Returns:
            Tuple of (response_a, response_b).

        Raises:
            Exception: If either participant fails.
        """
        # Run both generations in parallel (AC-14.2)
        response_a, response_b = await asyncio.gather(
            self._participant_a.generate(request),
            self._participant_b.generate(request),
        )

        return response_a, response_b

    # -------------------------------------------------------------------------
    # Phase 3: Reconciliation
    # -------------------------------------------------------------------------

    async def _reconcile(
        self,
        original_request: ChatCompletionRequest,
        content_a: str,
        content_b: str,
    ) -> ChatCompletionResponse:
        """Reconcile two outputs into a final synthesized answer.

        Creates a prompt containing both outputs and asks the reconciler
        to synthesize a final answer.

        Args:
            original_request: Original user request (for question context).
            content_a: Content from participant A.
            content_b: Content from participant B.

        Returns:
            ChatCompletionResponse from reconciler.
        """
        # Extract original question from request
        question = self._extract_user_question(original_request)

        # Build reconciliation prompt
        reconcile_user_content = RECONCILE_USER_PROMPT_TEMPLATE.format(
            question=question,
            response_a=content_a,
            response_b=content_b,
        )

        reconcile_request = ChatCompletionRequest(
            model=self.reconciler_model_id,
            messages=[
                Message(role="system", content=RECONCILE_SYSTEM_PROMPT),
                Message(role="user", content=reconcile_user_content),
            ],
            temperature=original_request.temperature,
            max_tokens=original_request.max_tokens,
        )

        return await self._reconciler.generate(reconcile_request)

    # -------------------------------------------------------------------------
    # Response Building
    # -------------------------------------------------------------------------

    def _build_response(
        self,
        reconciled_response: ChatCompletionResponse,
        response_a: ChatCompletionResponse,
        response_b: ChatCompletionResponse,
        agreement: float,
        start_time: float,
    ) -> ChatCompletionResponse:
        """Build final response with aggregated metadata.

        Args:
            reconciled_response: Response from reconciler.
            response_a: Response from participant A.
            response_b: Response from participant B.
            agreement: Agreement percentage between A and B.
            start_time: Start time for inference timing.

        Returns:
            ChatCompletionResponse with full metadata.
        """
        inference_time_ms = (time.perf_counter() - start_time) * 1000

        # Aggregate usage across all calls
        total_usage = self._aggregate_usage(
            response_a.usage, response_b.usage, reconciled_response.usage
        )

        # Collect all model IDs used
        models_used = self._collect_models_used(
            response_a, response_b, reconciled_response
        )

        return ChatCompletionResponse(
            id=f"chatcmpl-{uuid.uuid4().hex[:12]}",
            created=int(time.time()),
            model=reconciled_response.model,
            choices=reconciled_response.choices,
            usage=total_usage,
            orchestration=OrchestrationMetadata(
                mode="debate",
                models_used=models_used,
                total_inference_time_ms=inference_time_ms,
                agreement_score=agreement,
            ),
        )

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    def _extract_content(self, response: ChatCompletionResponse) -> str:
        """Extract content from response.

        Args:
            response: ChatCompletionResponse to extract from.

        Returns:
            Content string or empty string if not available.
        """
        if response.choices and response.choices[0].message.content:
            return response.choices[0].message.content
        return ""

    def _extract_user_question(self, request: ChatCompletionRequest) -> str:
        """Extract the user question from request messages.

        Args:
            request: ChatCompletionRequest to extract from.

        Returns:
            User question or empty string if not found.
        """
        for msg in reversed(request.messages):
            if msg.role == "user" and msg.content:
                return msg.content
        return ""

    def _aggregate_usage(
        self,
        usage_a: Usage | None,
        usage_b: Usage | None,
        usage_reconciler: Usage | None,
    ) -> Usage:
        """Aggregate usage stats from all responses.

        Args:
            usage_a: Usage from participant A.
            usage_b: Usage from participant B.
            usage_reconciler: Usage from reconciler.

        Returns:
            Aggregated Usage with totals.
        """
        total_prompt = 0
        total_completion = 0

        for usage in [usage_a, usage_b, usage_reconciler]:
            if usage:
                total_prompt += usage.prompt_tokens
                total_completion += usage.completion_tokens

        return Usage(
            prompt_tokens=total_prompt,
            completion_tokens=total_completion,
            total_tokens=total_prompt + total_completion,
        )

    def _collect_models_used(
        self,
        response_a: ChatCompletionResponse,
        response_b: ChatCompletionResponse,
        reconciled_response: ChatCompletionResponse,
    ) -> list[str]:
        """Collect unique model IDs used in all responses.

        Args:
            response_a: Response from participant A.
            response_b: Response from participant B.
            reconciled_response: Response from reconciler.

        Returns:
            List of unique model IDs.
        """
        models: list[str] = []

        for response in [response_a, response_b, reconciled_response]:
            if response.model and response.model not in models:
                models.append(response.model)

        return models
