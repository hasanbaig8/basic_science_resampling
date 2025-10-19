#!/usr/bin/env python3
"""
Generic intervention framework for CoT transcripts.

Provides abstract base class and implementations for intervening in
chain-of-thought reasoning at specified positions.
"""

from abc import ABC, abstractmethod
from typing import Optional
import re


class CoTIntervention(ABC):
    """
    Abstract base class for CoT interventions.

    An intervention takes a CoT transcript (starting from <think>) and returns
    a modified version with the intervention applied.
    """

    @abstractmethod
    def apply(self, cot_transcript: str) -> str:
        """
        Apply the intervention to a CoT transcript.

        Args:
            cot_transcript: The CoT reasoning text (including <think> tags)

        Returns:
            Modified transcript with intervention applied
        """
        pass


class DirectInsertionIntervention(CoTIntervention):
    """
    Insert text directly at a specified position in the CoT.

    This is the simplest intervention - it clips the CoT at a percentage
    through the content and inserts the intervention text.
    """

    def __init__(self, intervention_text: str, position_pct: float):
        """
        Initialize direct insertion intervention.

        Args:
            intervention_text: Text to insert
            position_pct: Position to insert (0.0-1.0, where 0.25=early, 0.5=mid, 0.75=late)
        """
        self.intervention_text = intervention_text
        self.position_pct = position_pct

        if not 0.0 <= position_pct <= 1.0:
            raise ValueError(f"position_pct must be between 0.0 and 1.0, got {position_pct}")

    def apply(self, cot_transcript: str) -> str:
        """
        Insert intervention text at specified position.

        Args:
            cot_transcript: Original CoT text with <think> tags

        Returns:
            Clipped CoT with intervention inserted
        """
        # Extract content between <think> tags
        think_match = re.search(r'<think>(.*?)</think>', cot_transcript, re.DOTALL)
        if not think_match:
            raise ValueError("No <think> tags found in transcript")

        think_content = think_match.group(1)

        # Calculate clip position (by character count)
        clip_position = int(len(think_content) * self.position_pct)

        # Clip the content
        clipped_content = think_content[:clip_position]

        # Add intervention text
        # Add newlines for formatting
        intervened_content = clipped_content + "\n\n" + self.intervention_text

        # Return with <think> tags (leave open for continuation)
        return f"<think>\n{intervened_content}\n"


class ParaphrasingIntervention(CoTIntervention):
    """
    Paraphrase intervention text based on context.

    This is a placeholder class. You can extend this to implement
    paraphrasing logic using Claude, Qwen, or other methods.
    """

    def __init__(
        self,
        intervention_text: str,
        position_pct: float,
        paraphrasing_fn: Optional[callable] = None
    ):
        """
        Initialize paraphrasing intervention.

        Args:
            intervention_text: Base text to paraphrase and insert
            position_pct: Position to insert (0.0-1.0)
            paraphrasing_fn: Optional function that takes (intervention_text, context) and returns paraphrased text
        """
        self.intervention_text = intervention_text
        self.position_pct = position_pct
        self.paraphrasing_fn = paraphrasing_fn

        if not 0.0 <= position_pct <= 1.0:
            raise ValueError(f"position_pct must be between 0.0 and 1.0, got {position_pct}")

    def apply(self, cot_transcript: str) -> str:
        """
        Paraphrase and insert intervention text at specified position.

        Args:
            cot_transcript: Original CoT text with <think> tags

        Returns:
            Clipped CoT with paraphrased intervention inserted
        """
        # Extract content between <think> tags
        think_match = re.search(r'<think>(.*?)</think>', cot_transcript, re.DOTALL)
        if not think_match:
            raise ValueError("No <think> tags found in transcript")

        think_content = think_match.group(1)

        # Calculate clip position
        clip_position = int(len(think_content) * self.position_pct)

        # Clip the content
        clipped_content = think_content[:clip_position]

        # Paraphrase the intervention text based on context
        if self.paraphrasing_fn is not None:
            paraphrased_text = self.paraphrasing_fn(self.intervention_text, clipped_content)
        else:
            # If no paraphrasing function provided, use original text
            paraphrased_text = self.intervention_text

        # Add paraphrased intervention
        intervened_content = clipped_content + "\n\n" + paraphrased_text

        # Return with <think> tags (leave open for continuation)
        return f"<think>\n{intervened_content}\n"


def extract_think_content(text: str) -> Optional[str]:
    """
    Extract content from <think> tags.

    Args:
        text: Text potentially containing <think> tags

    Returns:
        Content between <think> tags, or None if not found
    """
    match = re.search(r'<think>(.*?)</think>', text, re.DOTALL)
    if match:
        return match.group(1)
    return None


def clip_at_position(text: str, position_pct: float) -> str:
    """
    Clip text at a percentage position.

    Args:
        text: Text to clip
        position_pct: Position to clip at (0.0-1.0)

    Returns:
        Clipped text
    """
    clip_pos = int(len(text) * position_pct)
    return text[:clip_pos]
