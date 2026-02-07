"""Text Processing Utilities for LLM Response Cleaning.

Handles post-processing of LLM outputs, including stripping reasoning/thinking
tags from models like Qwen3, DeepSeek-R1, and other chain-of-thought models.

References:
    - Qwen3: Uses <think>...</think> tags for reasoning
    - DeepSeek-R1: Uses <think>...</think> tags
    - Some models use <reasoning>...</reasoning>
"""

from __future__ import annotations

import logging
import re
from typing import Final

logger = logging.getLogger(__name__)


# =============================================================================
# Reasoning Tag Patterns
# =============================================================================

# Models that emit reasoning tags (add new models here as discovered)
REASONING_TAG_PATTERNS: Final[list[tuple[str, re.Pattern[str]]]] = [
    # Qwen3 and DeepSeek-R1 style: <think>...</think>
    ("think", re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)),
    
    # Alternative reasoning tags
    ("reasoning", re.compile(r"<reasoning>.*?</reasoning>", re.DOTALL | re.IGNORECASE)),
    
    # Some models use <r>...</r> shorthand
    ("r", re.compile(r"<r>.*?</r>", re.DOTALL | re.IGNORECASE)),
    
    # Claude-style thinking (rare in local models)
    ("thinking", re.compile(r"<thinking>.*?</thinking>", re.DOTALL | re.IGNORECASE)),
    
    # Internal reasoning block
    ("internal_thought", re.compile(r"<internal_thought>.*?</internal_thought>", re.DOTALL | re.IGNORECASE)),
]

# Compiled combined pattern for efficiency (matches any reasoning tag)
# Matches both complete tags and unclosed opening tags at start
_COMBINED_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"<(?:think|thinking|reasoning|r|internal_thought)>.*?</(?:think|thinking|reasoning|r|internal_thought)>",
    re.DOTALL | re.IGNORECASE,
)

# Pattern for unclosed reasoning tags at the START of content
# This handles truncated responses where model starts thinking but gets cut off
_UNCLOSED_TAG_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"^\s*<(?:think|thinking|reasoning|r|internal_thought)>(?:(?!</(?:think|thinking|reasoning|r|internal_thought)>).)*$",
    re.DOTALL | re.IGNORECASE,
)


# =============================================================================
# Public API
# =============================================================================


def strip_reasoning_tags(content: str | None, preserve_final_answer: bool = True) -> str | None:
    """Strip reasoning/thinking tags from LLM response content.
    
    Removes chain-of-thought reasoning tags that models like Qwen3 and DeepSeek-R1
    emit. These tags contain internal reasoning that shouldn't be shown to users
    and consumes tokens without adding value to the final response.
    
    Args:
        content: The raw LLM response content (may contain reasoning tags).
        preserve_final_answer: If True, ensures content outside tags is preserved.
                               If False, returns empty string if entire content is tags.
    
    Returns:
        Content with reasoning tags removed and whitespace normalized.
        Returns None if input is None.
    
    Examples:
        >>> strip_reasoning_tags("<think>Let me think...</think>The answer is 42.")
        'The answer is 42.'
        
        >>> strip_reasoning_tags("<think>\\nAnalyzing...\\n</think>\\n\\nResult: success")
        'Result: success'
        
        >>> strip_reasoning_tags("No tags here, just text.")
        'No tags here, just text.'
    
    Reference:
        - Qwen3 emits <think>...</think> by default
        - DeepSeek-R1 uses same pattern
        - This runs in O(n) time with compiled regex
    """
    if content is None:
        return None
    
    if not content:
        return content
    
    # Fast path: check if any tags present before running regex
    if "<" not in content:
        return content
    
    # Strip all complete reasoning tag patterns first
    cleaned = _COMBINED_PATTERN.sub("", content)
    
    # Handle unclosed tags (truncated responses)
    # If the response starts with an unclosed reasoning tag and has no useful
    # content after stripping complete tags, the entire response is just thinking
    if _UNCLOSED_TAG_PATTERN.match(cleaned):
        logger.debug("Detected unclosed reasoning tag - response was truncated during thinking")
        # Strip the opening tag and return what we have (the thinking content)
        # This is still better than "[truncated]" since the model's thinking
        # often contains the answer approach even if not formally stated
        # Remove just the opening tag pattern at start
        unclosed_tag_start = re.match(
            r"^\s*<(?:think|thinking|reasoning|r|internal_thought)>\s*",
            cleaned,
            re.IGNORECASE
        )
        if unclosed_tag_start:
            cleaned = cleaned[unclosed_tag_start.end():]
            # If there's meaningful content after stripping tag, return it
            if cleaned.strip():
                logger.debug("Returning thinking content as response (truncated)")
                cleaned = cleaned.strip()
            else:
                cleaned = "[Response truncated]"
    
    # Normalize whitespace: collapse multiple newlines, strip leading/trailing
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)  # Max 2 consecutive newlines
    cleaned = cleaned.strip()
    
    # Log if we actually stripped something (at debug level)
    if cleaned != content.strip():
        original_len = len(content)
        cleaned_len = len(cleaned)
        stripped_len = original_len - cleaned_len
        logger.debug(
            "Stripped reasoning tags: %d chars removed (%d -> %d)",
            stripped_len,
            original_len,
            cleaned_len,
        )
    
    return cleaned


def extract_reasoning_content(content: str | None) -> dict[str, str | None]:
    """Extract both reasoning and final answer from content.
    
    Useful for debugging or when you want to preserve the reasoning separately.
    
    Args:
        content: The raw LLM response content.
    
    Returns:
        Dict with keys:
            - 'reasoning': The content inside reasoning tags (or None)
            - 'answer': The content outside reasoning tags
            - 'raw': The original content
    
    Example:
        >>> result = extract_reasoning_content("<think>Step 1...</think>Answer: 42")
        >>> result['reasoning']
        'Step 1...'
        >>> result['answer']
        'Answer: 42'
    """
    if content is None:
        return {"reasoning": None, "answer": None, "raw": None}
    
    # Find all reasoning blocks
    reasoning_parts = []
    for tag_name, pattern in REASONING_TAG_PATTERNS:
        matches = pattern.findall(content)
        for match in matches:
            # Extract just the inner content (without tags)
            inner_pattern = re.compile(
                rf"<{tag_name}>(.*?)</{tag_name}>",
                re.DOTALL | re.IGNORECASE,
            )
            inner_match = inner_pattern.search(match) if isinstance(match, str) else None
            if inner_match:
                reasoning_parts.append(inner_match.group(1).strip())
    
    reasoning = "\n---\n".join(reasoning_parts) if reasoning_parts else None
    answer = strip_reasoning_tags(content)
    
    return {
        "reasoning": reasoning,
        "answer": answer,
        "raw": content,
    }


def has_reasoning_tags(content: str | None) -> bool:
    """Check if content contains any reasoning tags.
    
    Args:
        content: The content to check.
    
    Returns:
        True if content contains reasoning tags, False otherwise.
    """
    if content is None or not content or "<" not in content:
        return False
    
    return bool(_COMBINED_PATTERN.search(content))
