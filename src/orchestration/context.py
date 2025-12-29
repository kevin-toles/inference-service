"""Context Management module for multi-model orchestration.

Provides:
- HandoffState: Structured state for model-to-model handoffs
- MODEL_CONTEXT_BUDGETS: Token budget allocation per model
- fit_to_budget(): Iterative compression to fit context window
- inject_trajectory(): Trajectory injection for prompts
- ErrorContaminationDetector: Detect error contamination in handoffs

Per ARCHITECTURE.md Context Management section.
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, ClassVar, Protocol


if TYPE_CHECKING:
    from src.providers.base import InferenceProvider


# =============================================================================
# Exceptions
# =============================================================================


class CompressionFailedError(Exception):
    """Raised when content cannot be compressed to fit budget."""

    pass


# =============================================================================
# Data Structures
# =============================================================================


@dataclass
class HandoffState:
    """Structured state for model-to-model handoffs.

    Note: Uses field(default_factory=list) per CODING_PATTERNS_ANALYSIS AP-1.5
    to avoid mutable default argument anti-pattern.
    """

    request_id: str
    goal: str  # Original user intent
    current_step: int  # Position in pipeline
    total_steps: int

    # Mutable fields use default_factory (AP-1.5 compliance)
    constraints: list[str] = field(default_factory=list)
    decisions_made: list[str] = field(default_factory=list)
    evidence_refs: list[str] = field(default_factory=list)
    active_errors: list[str] = field(default_factory=list)
    resolved_errors: list[str] = field(default_factory=list)

    # Optional fields explicitly annotated (AP-1.1 compliance)
    compressed_context: str | None = None


@dataclass
class ErrorIssue:
    """Represents a detected error issue during handoff validation."""

    type: str
    severity: str
    marker: str | None = None
    decision: str | None = None
    count: int | None = None


@dataclass
class ValidationResult:
    """Result of handoff validation."""

    valid: bool
    issues: list[ErrorIssue]
    recommendation: str


# =============================================================================
# Context Budget Allocation
# =============================================================================


MODEL_CONTEXT_BUDGETS: dict[str, dict[str, int]] = {
    "llama-3.2-3b": {
        "total": 8192,
        "system": 500,
        "trajectory": 300,
        "handoff": 1500,
        "user_query": 2000,
        "generation": 3892,
    },
    "granite-8b-code-128k": {
        "total": 131072,
        "system": 1000,
        "trajectory": 500,
        "handoff": 4000,
        "user_query": 32000,
        "generation": 93572,
    },
    "qwen2.5-7b": {
        "total": 32768,
        "system": 800,
        "trajectory": 400,
        "handoff": 3000,
        "user_query": 8000,
        "generation": 20568,
    },
    "deepseek-r1-7b": {
        "total": 32768,
        "system": 800,
        "trajectory": 400,
        "handoff": 3000,
        "user_query": 8000,
        "generation": 20568,
    },
    "phi-4": {
        "total": 16384,
        "system": 600,
        "trajectory": 350,
        "handoff": 2000,
        "user_query": 4000,
        "generation": 9434,
    },
    "phi-3-medium-128k": {
        "total": 131072,
        "system": 1000,
        "trajectory": 500,
        "handoff": 4000,
        "user_query": 32000,
        "generation": 93572,
    },
    "gpt-oss-20b": {
        "total": 32768,
        "system": 800,
        "trajectory": 400,
        "handoff": 3000,
        "user_query": 8000,
        "generation": 20568,
    },
    "granite-20b-code": {
        "total": 8192,
        "system": 500,
        "trajectory": 300,
        "handoff": 1500,
        "user_query": 2000,
        "generation": 3892,
    },
}


# =============================================================================
# Token Counting Helpers
# =============================================================================


class TokenizableModel(Protocol):
    """Protocol for models that support tokenization."""

    def tokenize(self, text: bytes) -> list[int]:
        """Tokenize text and return token IDs."""
        ...


def _count_tokens(content: str, model: TokenizableModel) -> int:
    """Count tokens in content.

    Args:
        content: Text content to count tokens for.
        model: Model with tokenize method.

    Returns:
        Number of tokens in content.
    """
    return len(model.tokenize(content.encode()))


def _calculate_compression_ratio(target: int, current: int) -> float:
    """Calculate required compression ratio with safety margin.

    Args:
        target: Target number of tokens.
        current: Current number of tokens.

    Returns:
        Compression ratio with 10% safety margin.
    """
    return (target / current) * 0.9  # 10% safety margin


async def _apply_compression(
    content: str,
    target_ratio: float,
    preserve: list[str] | None = None,
    drop: list[str] | None = None,
    provider: "InferenceProvider | None" = None,
) -> str:
    """Apply compression using any loaded LLM via InferenceProvider.

    This function uses the provided InferenceProvider to perform LLM-based
    compression. Any loaded model can be used for compression, making this
    solution flexible and provider-agnostic.

    Args:
        content: Content to compress.
        target_ratio: Target compression ratio (0.0-1.0).
        preserve: Categories to preserve (e.g., ["decisions", "constraints"]).
        drop: Categories to drop (e.g., ["reasoning_chains", "examples"]).
        provider: InferenceProvider to use for LLM-based compression.
                  If None, falls back to simple truncation.

    Returns:
        Compressed content.

    Note:
        - Uses `preserve or []` pattern per AP-1.5 for mutable defaults.
        - Provider parameter allows any loaded LLM to compress.
    """
    # AP-1.5: Handle None defaults for mutable arguments
    preserve_list = preserve or []
    drop_list = drop or []

    # Calculate target length
    target_len = int(len(content) * target_ratio)

    # If no provider, fall back to simple truncation (legacy behavior)
    if provider is None:
        return content[:target_len]

    # Build compression prompt for LLM
    from src.models.requests import ChatCompletionRequest, Message

    preserve_text = ", ".join(preserve_list) if preserve_list else "all key information"
    drop_text = ", ".join(drop_list) if drop_list else "redundant details"

    compression_prompt = f"""Compress the following content to approximately {target_len} characters (target ratio: {target_ratio:.0%}).

PRESERVE: {preserve_text}
DROP/REMOVE: {drop_text}

Return ONLY the compressed content, no explanations.

CONTENT TO COMPRESS:
{content}"""

    request = ChatCompletionRequest(
        model=provider.model_info.model_id,
        messages=[
            Message(role="system", content="You are a compression assistant. Compress content while preserving key information."),
            Message(role="user", content=compression_prompt),
        ],
        max_tokens=target_len,  # Limit output to target length
        temperature=0.1,  # Low temperature for deterministic compression
    )

    response = await provider.generate(request)

    # Extract compressed content from response
    if response.choices and response.choices[0].message:
        return response.choices[0].message.content or content[:target_len]

    # Fallback to truncation if LLM fails
    return content[:target_len]


# =============================================================================
# Fit to Budget
# =============================================================================


async def fit_to_budget(
    content: str,
    max_tokens: int,
    model: TokenizableModel,
    max_iterations: int = 3,
    provider: "InferenceProvider | None" = None,
) -> str:
    """Iteratively compress until under budget.

    Note: Uses iterative approach per CODING_PATTERNS_ANALYSIS AP-2.1
    to avoid high cognitive complexity from recursion.

    Args:
        content: Content to fit within budget.
        max_tokens: Maximum number of tokens allowed.
        model: Model for tokenization.
        max_iterations: Maximum compression iterations.
        provider: InferenceProvider to use for LLM-based compression.
                  If None, falls back to simple truncation.

    Returns:
        Content that fits within token budget.

    Raises:
        CompressionFailedError: If cannot compress to budget after max iterations.
    """
    for _iteration in range(max_iterations):
        current_tokens = _count_tokens(content, model)

        if current_tokens <= max_tokens:
            return content

        # Calculate compression ratio needed
        ratio = _calculate_compression_ratio(max_tokens, current_tokens)

        # Compress using provider (LLM-based) or fallback to truncation
        content = await _apply_compression(
            content=content,
            target_ratio=ratio,
            preserve=["decisions", "constraints", "errors"],
            drop=["reasoning_chains", "examples", "verbose"],
            provider=provider,
        )

    # Max iterations reached - raise instead of silent truncation
    raise CompressionFailedError(
        f"Could not compress to {max_tokens} tokens after {max_iterations} iterations"
    )


# =============================================================================
# Trajectory Injection
# =============================================================================


def inject_trajectory(
    state: HandoffState,
    step_name: str,
    previous_decision: str,
    next_task: str,
    forbidden: list[str],
) -> str:
    """Inject trajectory into prompt to prevent 'lost agent' problem.

    Every prompt includes trajectory per ARCHITECTURE.md specification.

    Args:
        state: Current handoff state.
        step_name: Name of the current step.
        previous_decision: What was decided in previous step.
        next_task: What this model must decide.
        forbidden: List of forbidden actions.

    Returns:
        Formatted trajectory string for prompt injection.
    """
    forbidden_text = "\n".join(f"- {f}" for f in forbidden) if forbidden else "None"

    return f"""## Current Position
- Goal: {state.goal}
- Step: {state.current_step}/{state.total_steps} - {step_name}
- Previous: {previous_decision}
- Next: {next_task}
- Forbidden:
{forbidden_text}
"""


# =============================================================================
# Error Contamination Detection
# =============================================================================


class ErrorContaminationDetector:
    """Detect and quarantine error-contaminated context between pipeline steps.

    Per ARCHITECTURE.md: "Errors are sticky - once incorrect assumptions enter
    context, they bias subsequent reasoning."
    """

    ERROR_MARKERS: ClassVar[list[str]] = [
        "I apologize",
        "I made an error",
        "That's incorrect",
        "hallucination",
        "I don't have information about",
    ]

    def validate_handoff(
        self, state: HandoffState, output: str
    ) -> ValidationResult:
        """Check for error contamination before passing to next model.

        Args:
            state: Current handoff state.
            output: Output from current model to validate.

        Returns:
            ValidationResult indicating if handoff is safe to proceed.
        """
        issues: list[ErrorIssue] = []

        # Check for error markers in output
        for marker in self.ERROR_MARKERS:
            if marker.lower() in output.lower():
                issues.append(
                    ErrorIssue(
                        type="error_marker_detected",
                        marker=marker,
                        severity="warning",
                    )
                )

        # Check for contradiction with previous decisions
        for decision in state.decisions_made:
            if self._contradicts(output, decision):
                issues.append(
                    ErrorIssue(
                        type="contradiction_detected",
                        decision=decision,
                        severity="high",
                    )
                )

        # Check if error count is growing
        if len(state.active_errors) > state.current_step:
            issues.append(
                ErrorIssue(
                    type="error_accumulation",
                    count=len(state.active_errors),
                    severity="high",
                )
            )

        # Determine if valid based on high severity issues
        high_severity_count = len([i for i in issues if i.severity == "high"])
        valid = high_severity_count == 0

        return ValidationResult(
            valid=valid,
            issues=issues,
            recommendation="proceed" if valid else "quarantine",
        )

    def _contradicts(self, output: str, decision: str) -> bool:
        """Check if output contradicts a previous decision.

        This is a simplified implementation. A production version would use
        semantic similarity or an LLM to detect contradictions.

        Args:
            output: Current model output.
            decision: Previous decision to check against.

        Returns:
            True if contradiction detected, False otherwise.
        """
        # Simple negation detection - production would be more sophisticated
        negation_patterns = [
            f"not {decision.lower()}",
            f"don't {decision.lower()}",
            f"shouldn't {decision.lower()}",
            f"instead of {decision.lower()}",
        ]

        output_lower = output.lower()
        return any(pattern in output_lower for pattern in negation_patterns)
