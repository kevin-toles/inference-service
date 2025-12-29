"""Ensemble mode orchestration - All models generate → Consensus → Synthesize.

EnsembleMode implements multi-model orchestration where:
1. All participants generate responses in parallel
2. Consensus score is calculated across all outputs
3. Synthesizer creates final answer from agreed points
4. Disagreements are identified and flagged

Flow:
    Request → [All](parallel) → Consensus → Synthesize → Response

Per ARCHITECTURE.md Orchestration Modes (ensemble):
- ensemble: All vote, synthesize
- Min Models: 2+
- Flow: Request → [All](parallel) → Consensus → Response

Reference: WBS-INF15 AC-15.1, AC-15.2, AC-15.3, AC-15.4
"""

import asyncio
import time
import uuid
from collections import Counter

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


DEFAULT_MIN_AGREEMENT = 0.7

SYNTHESIS_SYSTEM_PROMPT = """You are synthesizing a final answer from multiple model responses.

Your task is to:
1. Identify points where most models agree (consensus points)
2. Create a coherent answer that emphasizes these consensus points
3. If there are significant disagreements, note them briefly
4. Provide the most accurate and complete answer based on the majority opinion

Do NOT mention that you are synthesizing from multiple responses.
Simply provide the best possible answer."""

SYNTHESIS_USER_PROMPT_TEMPLATE = """Original question: {question}

{responses_section}

Please synthesize a final answer based on the consensus among these responses.
Consensus score: {consensus_score:.1%}"""


# =============================================================================
# Consensus & Disagreement Functions
# =============================================================================


def calculate_pairwise_consensus(responses: list[str]) -> float:
    """Calculate consensus score using pairwise word overlap (Jaccard similarity).

    Computes average Jaccard similarity across all unique pairs of responses.

    Args:
        responses: List of response strings to compare.

    Returns:
        Consensus score between 0.0 and 1.0.

    Example:
        >>> calculate_pairwise_consensus(["Paris is capital", "Paris is capital"])
        1.0
    """
    if len(responses) <= 1:
        return 1.0  # Single or no response = perfect consensus

    # Calculate pairwise Jaccard similarities
    similarities: list[float] = []

    for i in range(len(responses)):
        for j in range(i + 1, len(responses)):
            sim = _jaccard_similarity(responses[i], responses[j])
            similarities.append(sim)

    if not similarities:
        return 1.0

    return sum(similarities) / len(similarities)


def _jaccard_similarity(text_a: str, text_b: str) -> float:
    """Calculate Jaccard similarity between two texts.

    Args:
        text_a: First text.
        text_b: Second text.

    Returns:
        Similarity score between 0.0 and 1.0.
    """
    if not text_a and not text_b:
        return 1.0
    if not text_a or not text_b:
        return 0.0

    words_a = set(text_a.lower().split())
    words_b = set(text_b.lower().split())

    intersection = words_a & words_b
    union = words_a | words_b

    if not union:
        return 1.0

    return len(intersection) / len(union)


def extract_disagreement_points(responses: list[str]) -> list[str]:
    """Extract points where responses disagree.

    Identifies words/terms that appear in a minority of responses,
    suggesting disagreement.

    Args:
        responses: List of response strings.

    Returns:
        List of disagreement descriptions.

    Example:
        >>> extract_disagreement_points(["Paris", "Paris", "Lyon"])
        ["'Lyon' appears in minority (1/3 responses)"]
    """
    if len(responses) <= 1:
        return []

    # Common stopwords to skip
    stopwords = {"the", "is", "a", "an", "and", "or", "in", "on", "at", "to", "of"}

    # Count word occurrences across all responses
    word_counts: Counter[str] = Counter()
    word_sources: dict[str, set[int]] = {}

    for i, response in enumerate(responses):
        # Tokenize: split and clean punctuation
        raw_words = response.split()
        clean_words: set[str] = set()
        for w in raw_words:
            # Strip punctuation from word boundaries
            cleaned = w.strip(".,!?;:'\"()-")
            if cleaned:
                clean_words.add(cleaned.lower())

        for word in clean_words:
            word_counts[word] += 1
            if word not in word_sources:
                word_sources[word] = set()
            word_sources[word].add(i)

    # Find words that appear in minority of responses
    disagreements: list[str] = []
    total_responses = len(responses)
    majority_threshold = total_responses / 2

    for word, count in word_counts.items():
        # Skip very short common words (but keep numbers like "7" or "42")
        if word in stopwords:
            continue

        # If word appears in less than half of responses, it's a potential disagreement
        if 0 < count < majority_threshold:
            disagreements.append(
                f"'{word}' appears in minority ({count}/{total_responses} responses)"
            )

    return disagreements


# =============================================================================
# EnsembleMode Implementation
# =============================================================================


class EnsembleMode:
    """Ensemble mode orchestration - All generate → Consensus → Synthesize.

    Coordinates multiple models in a parallel-vote-synthesize pattern:
    1. All participants generate responses in parallel using asyncio.gather
    2. Consensus score is calculated using pairwise Jaccard similarity
    3. Synthesizer creates final answer from all outputs
    4. Disagreements are identified and can be reported

    AC-15.1: All models generate in parallel
    AC-15.2: Calculates consensus score
    AC-15.3: Synthesizes from agreed points
    AC-15.4: Flags disagreements

    Attributes:
        participants: List of participant providers.
        synthesizer: Provider for synthesis (often same as primary participant).
        min_agreement: Minimum consensus threshold (default 0.7).

    Example:
        model_a = LlamaCppProvider(model_path="phi-4.gguf")
        model_b = LlamaCppProvider(model_path="deepseek-r1-7b.gguf")
        model_c = LlamaCppProvider(model_path="qwen2.5-7b.gguf")
        mode = EnsembleMode(
            participants=[model_a, model_b, model_c],
            synthesizer=model_a,  # phi-4 synthesizes
            min_agreement=0.7,
        )
        response = await mode.execute(request)
    """

    def __init__(
        self,
        participants: list[InferenceProvider],
        synthesizer: InferenceProvider,
        min_agreement: float = DEFAULT_MIN_AGREEMENT,
    ) -> None:
        """Initialize EnsembleMode with participant and synthesizer providers.

        Args:
            participants: List of participant providers (min 2).
            synthesizer: Provider for synthesis. Can be one of the participants.
            min_agreement: Minimum consensus score threshold. Default 0.7.
        """
        self._participants = participants
        self._synthesizer = synthesizer
        self._min_agreement = min_agreement

    # -------------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------------

    @property
    def participants(self) -> list[InferenceProvider]:
        """Get list of participant providers."""
        return self._participants

    @property
    def synthesizer(self) -> InferenceProvider:
        """Get synthesizer provider."""
        return self._synthesizer

    @property
    def min_agreement(self) -> float:
        """Get minimum agreement threshold."""
        return self._min_agreement

    @property
    def participant_model_ids(self) -> list[str]:
        """Get model IDs for all participants."""
        return [p.model_info.model_id for p in self._participants]

    @property
    def synthesizer_model_id(self) -> str:
        """Get model ID for synthesizer."""
        return self._synthesizer.model_info.model_id

    # -------------------------------------------------------------------------
    # Main Execution
    # -------------------------------------------------------------------------

    async def execute(
        self, request: ChatCompletionRequest
    ) -> ChatCompletionResponse:
        """Execute ensemble mode orchestration.

        Steps:
        1. Generate responses from all participants in parallel
        2. Calculate consensus score across all outputs
        3. Call synthesizer to create final answer
        4. Build response with aggregated metadata

        Args:
            request: Original chat completion request.

        Returns:
            ChatCompletionResponse with synthesized content and metadata.

        Raises:
            Exception: If any participant fails during generation.
        """
        start_time = time.perf_counter()

        # Phase 1: Parallel generation using asyncio.gather (AC-15.1)
        responses = await self._parallel_generate(request)

        # Extract content from all responses
        contents = [self._extract_content(r) for r in responses]

        # Phase 2: Calculate consensus (AC-15.2)
        consensus = calculate_pairwise_consensus(contents)

        # Phase 3: Identify disagreements (AC-15.4)
        disagreements = extract_disagreement_points(contents)

        # Phase 4: Synthesize (AC-15.3)
        synthesized_response = await self._synthesize(
            request, contents, consensus
        )

        # Build final response with metadata
        return self._build_response(
            synthesized_response=synthesized_response,
            participant_responses=responses,
            consensus=consensus,
            disagreements=disagreements,
            start_time=start_time,
        )

    # -------------------------------------------------------------------------
    # Phase 1: Parallel Generation
    # -------------------------------------------------------------------------

    async def _parallel_generate(
        self, request: ChatCompletionRequest
    ) -> list[ChatCompletionResponse]:
        """Generate responses from all participants in parallel.

        Uses asyncio.gather to run all generations concurrently.

        Args:
            request: Original request to send to all participants.

        Returns:
            List of responses from all participants.

        Raises:
            Exception: If any participant fails.
        """
        # Run all generations in parallel (AC-15.1)
        tasks = [p.generate(request) for p in self._participants]
        responses = await asyncio.gather(*tasks)
        return list(responses)

    # -------------------------------------------------------------------------
    # Phase 4: Synthesis
    # -------------------------------------------------------------------------

    async def _synthesize(
        self,
        original_request: ChatCompletionRequest,
        contents: list[str],
        consensus: float,
    ) -> ChatCompletionResponse:
        """Synthesize final answer from all participant outputs.

        Creates a prompt containing all outputs and asks the synthesizer
        to create a final answer based on consensus.

        Args:
            original_request: Original user request (for question context).
            contents: List of content strings from all participants.
            consensus: Calculated consensus score.

        Returns:
            ChatCompletionResponse from synthesizer.
        """
        # Extract original question from request
        question = self._extract_user_question(original_request)

        # Build responses section
        responses_section = self._format_responses_section(contents)

        # Build synthesis prompt
        synthesis_user_content = SYNTHESIS_USER_PROMPT_TEMPLATE.format(
            question=question,
            responses_section=responses_section,
            consensus_score=consensus,
        )

        synthesis_request = ChatCompletionRequest(
            model=self.synthesizer_model_id,
            messages=[
                Message(role="system", content=SYNTHESIS_SYSTEM_PROMPT),
                Message(role="user", content=synthesis_user_content),
            ],
            temperature=original_request.temperature,
            max_tokens=original_request.max_tokens,
        )

        return await self._synthesizer.generate(synthesis_request)

    # -------------------------------------------------------------------------
    # Response Building
    # -------------------------------------------------------------------------

    def _build_response(
        self,
        synthesized_response: ChatCompletionResponse,
        participant_responses: list[ChatCompletionResponse],
        consensus: float,
        disagreements: list[str],
        start_time: float,
    ) -> ChatCompletionResponse:
        """Build final response with aggregated metadata.

        Args:
            synthesized_response: Response from synthesizer.
            participant_responses: Responses from all participants.
            consensus: Consensus score.
            disagreements: List of identified disagreements.
            start_time: Start time for inference timing.

        Returns:
            ChatCompletionResponse with full metadata.
        """
        inference_time_ms = (time.perf_counter() - start_time) * 1000

        # Aggregate usage across all calls
        total_usage = self._aggregate_usage(
            participant_responses, synthesized_response.usage
        )

        # Collect all model IDs used
        models_used = self._collect_models_used(
            participant_responses, synthesized_response
        )

        return ChatCompletionResponse(
            id=f"chatcmpl-{uuid.uuid4().hex[:12]}",
            created=int(time.time()),
            model=synthesized_response.model,
            choices=synthesized_response.choices,
            usage=total_usage,
            orchestration=OrchestrationMetadata(
                mode="ensemble",
                models_used=models_used,
                total_inference_time_ms=inference_time_ms,
                agreement_score=consensus,
                disagreement_points=disagreements if disagreements else None,
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

    def _format_responses_section(self, contents: list[str]) -> str:
        """Format all responses into a numbered section.

        Args:
            contents: List of response content strings.

        Returns:
            Formatted string with numbered responses.
        """
        lines: list[str] = []
        for i, content in enumerate(contents, 1):
            lines.append(f"Response {i}:")
            lines.append(content)
            lines.append("")
        return "\n".join(lines)

    def _aggregate_usage(
        self,
        participant_responses: list[ChatCompletionResponse],
        synthesizer_usage: Usage | None,
    ) -> Usage:
        """Aggregate usage stats from all responses.

        Args:
            participant_responses: Responses from all participants.
            synthesizer_usage: Usage from synthesizer.

        Returns:
            Aggregated Usage with totals.
        """
        total_prompt = 0
        total_completion = 0

        for response in participant_responses:
            if response.usage:
                total_prompt += response.usage.prompt_tokens
                total_completion += response.usage.completion_tokens

        if synthesizer_usage:
            total_prompt += synthesizer_usage.prompt_tokens
            total_completion += synthesizer_usage.completion_tokens

        return Usage(
            prompt_tokens=total_prompt,
            completion_tokens=total_completion,
            total_tokens=total_prompt + total_completion,
        )

    def _collect_models_used(
        self,
        participant_responses: list[ChatCompletionResponse],
        synthesized_response: ChatCompletionResponse,
    ) -> list[str]:
        """Collect unique model IDs used in all responses.

        Args:
            participant_responses: Responses from all participants.
            synthesized_response: Response from synthesizer.

        Returns:
            List of unique model IDs.
        """
        models: list[str] = []

        for response in participant_responses:
            if response.model and response.model not in models:
                models.append(response.model)

        if synthesized_response.model and synthesized_response.model not in models:
            models.append(synthesized_response.model)

        return models
