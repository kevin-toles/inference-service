"""Critique mode orchestration - Generator → Critic → Revise flow.

CritiqueMode implements multi-model orchestration where:
1. Generator (Model A) creates initial response
2. Critic (Model B) reviews for issues
3. Generator (Model A) revises based on feedback

Flow:
    Request → A(gen) → B(critique) → A(revise) → Response

Per ARCHITECTURE.md Orchestration Modes (critique):
- critique: Generate then critique
- Min Models: 2
- Flow: Request → A(gen) → B(critique) → A(revise) → Response

Reference: WBS-INF12 AC-12.1, AC-12.2, AC-12.3, AC-12.4
"""

import time

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


DEFAULT_MAX_ROUNDS = 3
NO_ISSUES_MARKER = "NO ISSUES FOUND"
ISSUES_MARKER = "ISSUES"

CRITIQUE_SYSTEM_PROMPT = """You are a critical reviewer. Analyze the following response for:
1. Factual accuracy and hallucinations
2. Missing information or edge cases
3. Clarity and completeness
4. Logical errors or inconsistencies

If you find issues, respond with:
ISSUES FOUND:
- [List each issue]
RECOMMENDATION: [How to fix]

If the response is satisfactory, respond with:
NO ISSUES FOUND. Response is complete and accurate."""

REVISE_SYSTEM_PROMPT = """You are revising your previous response based on feedback.
Incorporate the critique feedback to improve your response.
Address all identified issues while maintaining the original intent."""


# =============================================================================
# CritiqueMode Implementation
# =============================================================================


class CritiqueMode:
    """Critique mode orchestration - Generator → Critic → Revise.

    Coordinates two models in a critique-revise loop:
    1. Generator creates initial response
    2. Critic reviews and identifies issues
    3. Generator revises based on critique (if issues found)
    4. Repeat until no issues or max_rounds reached

    AC-12.1: Generator → Critic → Revise flow
    AC-12.2: Uses HandoffState between steps
    AC-12.3: Respects max_rounds setting
    AC-12.4: Reports models_used in metadata

    Attributes:
        generator: Provider for generation/revision (Model A).
        critic: Provider for critique (Model B).
        max_rounds: Maximum critique-revise iterations.

    Example:
        generator = LlamaCppProvider(model_path="phi-4.gguf")
        critic = LlamaCppProvider(model_path="deepseek-r1-7b.gguf")
        mode = CritiqueMode(generator=generator, critic=critic, max_rounds=3)
        response = await mode.execute(request)
    """

    def __init__(
        self,
        generator: InferenceProvider,
        critic: InferenceProvider,
        max_rounds: int = DEFAULT_MAX_ROUNDS,
    ) -> None:
        """Initialize CritiqueMode with generator and critic providers.

        Args:
            generator: Provider for generation/revision (Model A).
            critic: Provider for critique (Model B).
            max_rounds: Maximum critique-revise iterations. Default 3.
        """
        self._generator = generator
        self._critic = critic
        self._max_rounds = max_rounds

    @property
    def generator(self) -> InferenceProvider:
        """Get the generator provider."""
        return self._generator

    @property
    def critic(self) -> InferenceProvider:
        """Get the critic provider."""
        return self._critic

    @property
    def max_rounds(self) -> int:
        """Get the maximum rounds setting."""
        return self._max_rounds

    @property
    def generator_model_id(self) -> str:
        """Get the generator model ID."""
        return self._generator.model_info.model_id

    @property
    def critic_model_id(self) -> str:
        """Get the critic model ID."""
        return self._critic.model_info.model_id

    async def execute(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        """Execute critique mode orchestration.

        Flow:
        1. Generate initial response
        2. If max_rounds > 0, critique the response
        3. If issues found and rounds remaining, revise and repeat
        4. Return final response with orchestration metadata

        AC-12.1: Implements Generator → Critic → Revise flow.
        AC-12.2: Passes content between phases via request messages.
        AC-12.3: Respects max_rounds setting.
        AC-12.4: Reports models_used in metadata.

        Args:
            request: Chat completion request.

        Returns:
            ChatCompletionResponse with orchestration metadata.

        Raises:
            Any errors from providers are propagated.
        """
        start_time = time.perf_counter()

        # Track usage across all calls
        total_prompt_tokens = 0
        total_completion_tokens = 0

        # Phase 1: Generate initial response
        generation_response = await self._generator.generate(request)
        total_prompt_tokens += generation_response.usage.prompt_tokens
        total_completion_tokens += generation_response.usage.completion_tokens

        current_content = self._extract_content(generation_response)
        final_response = generation_response
        rounds_completed = 0
        issues_remain = False

        # Phase 2-3: Critique and Revise loop
        for round_num in range(self._max_rounds):
            # Critique phase
            critique_request = self._build_critique_request(request, current_content)
            critique_response = await self._critic.generate(critique_request)
            total_prompt_tokens += critique_response.usage.prompt_tokens
            total_completion_tokens += critique_response.usage.completion_tokens

            critique_content = self._extract_content(critique_response)
            issues_found = self._has_issues(critique_content)

            if not issues_found:
                # No issues - we're done
                issues_remain = False
                break

            # Issues found - need to revise
            rounds_completed = round_num + 1
            issues_remain = True

            # Revise phase - always revise when issues found
            revise_request = self._build_revise_request(
                request, current_content, critique_content
            )
            revise_response = await self._generator.generate(revise_request)
            total_prompt_tokens += revise_response.usage.prompt_tokens
            total_completion_tokens += revise_response.usage.completion_tokens

            current_content = self._extract_content(revise_response)
            final_response = revise_response

        # Calculate final score
        final_score = self._calculate_score(issues_remain, rounds_completed)

        # Calculate total inference time
        inference_time_ms = (time.perf_counter() - start_time) * 1000

        # Build final response with orchestration metadata
        models_used = [self.generator_model_id, self.critic_model_id]
        if self._generator == self._critic:
            # Same model for both roles
            models_used = [self.generator_model_id]

        final_response.orchestration = OrchestrationMetadata(
            mode="critique",
            models_used=models_used,
            total_inference_time_ms=inference_time_ms,
            rounds=rounds_completed,
            final_score=final_score,
            agreement_score=None,  # Not applicable for critique mode
        )

        # Update usage to aggregate all calls
        final_response.usage = Usage(
            prompt_tokens=total_prompt_tokens,
            completion_tokens=total_completion_tokens,
            total_tokens=total_prompt_tokens + total_completion_tokens,
        )

        return final_response

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

    def _has_issues(self, critique_content: str) -> bool:
        """Check if critique found issues.

        Args:
            critique_content: Critique response content.

        Returns:
            True if issues were found, False otherwise.
        """
        # Check for no issues marker
        if NO_ISSUES_MARKER in critique_content.upper():
            return False
        # Check for issues marker
        if ISSUES_MARKER in critique_content.upper():
            return True
        # Default: assume issues if not explicitly clean
        return True

    def _build_critique_request(
        self, original_request: ChatCompletionRequest, content: str
    ) -> ChatCompletionRequest:
        """Build request for critique phase.

        Args:
            original_request: Original user request.
            content: Generated content to critique.

        Returns:
            Request for critic model.
        """
        # Build critique request with system prompt and content to review
        messages = [
            Message(role="system", content=CRITIQUE_SYSTEM_PROMPT),
            Message(
                role="user",
                content=f"Original request: {self._extract_user_message(original_request)}\n\n"
                f"Response to review:\n{content}",
            ),
        ]

        return ChatCompletionRequest(
            model=self.critic_model_id,
            messages=messages,
            temperature=original_request.temperature,
            max_tokens=original_request.max_tokens,
        )

    def _build_revise_request(
        self,
        original_request: ChatCompletionRequest,
        content: str,
        critique: str,
    ) -> ChatCompletionRequest:
        """Build request for revision phase.

        Args:
            original_request: Original user request.
            content: Current content to revise.
            critique: Critique feedback to incorporate.

        Returns:
            Request for generator model to revise.
        """
        # Build revision request with original context and critique
        messages = list(original_request.messages)  # Copy original messages

        # Add revision instruction
        messages.append(
            Message(
                role="user",
                content=f"{REVISE_SYSTEM_PROMPT}\n\n"
                f"Your previous response:\n{content}\n\n"
                f"Critique feedback:\n{critique}\n\n"
                "Please provide an improved response addressing the feedback.",
            )
        )

        return ChatCompletionRequest(
            model=self.generator_model_id,
            messages=messages,
            temperature=original_request.temperature,
            max_tokens=original_request.max_tokens,
        )

    def _extract_user_message(self, request: ChatCompletionRequest) -> str:
        """Extract the main user message from request.

        Args:
            request: Chat completion request.

        Returns:
            User message content.
        """
        for message in reversed(request.messages):
            if message.role == "user" and message.content:
                return message.content
        return ""

    def _calculate_score(self, issues_remain: bool, rounds_completed: int) -> float:
        """Calculate final quality score.

        Args:
            issues_remain: Whether issues still exist after critique.
            rounds_completed: Number of critique-revise rounds completed.

        Returns:
            Score between 0.0 and 1.0.
        """
        if not issues_remain:
            return 1.0

        # Lower score if issues remain after max rounds
        # More rounds completed = better (we tried more)
        # But still <1.0 because issues remain
        base_score = 0.5
        round_bonus = min(rounds_completed * 0.1, 0.4)  # Max 0.4 bonus
        return base_score + round_bonus
