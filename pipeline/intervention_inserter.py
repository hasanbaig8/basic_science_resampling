#!/usr/bin/env python3
"""
InterventionInserter - Clip rollouts and insert intervention text.

This is the main component that will evolve as we experiment with
different intervention strategies. Currently supports direct insertion.
"""

import re
from typing import Optional
from abc import ABC, abstractmethod


class InterventionStrategy(ABC):
    """
    Abstract base class for intervention strategies.

    This allows for easy extension with different intervention approaches
    (e.g., paraphrasing, contextual insertion, etc.)
    """

    @abstractmethod
    def apply(self, rollout: str, intervention_text: str, position_pct: float) -> str:
        """
        Apply the intervention to a rollout.

        Args:
            rollout: The original rollout text (may include <think> tags)
            intervention_text: Text to insert
            position_pct: Where to clip and insert (0.0-1.0)

        Returns:
            Modified text ready for continuation (with open <think> tag if applicable)
        """
        pass


class DirectInsertionStrategy(InterventionStrategy):
    """
    Simple strategy: clip at position and insert text directly.

    This is the baseline intervention approach.
    """

    def apply(self, rollout: str, intervention_text: str, position_pct: float) -> str:
        """
        Clip rollout and insert intervention text.

        If rollout has <think> tags, extracts content and preserves structure.
        Otherwise, clips the raw text.

        Args:
            rollout: The original rollout text
            intervention_text: Text to insert after clipping
            position_pct: Position to clip at (0.0-1.0)

        Returns:
            Clipped text with intervention inserted, ending with open <think> tag
        """
        if not 0.0 <= position_pct <= 1.0:
            raise ValueError(f"position_pct must be between 0.0 and 1.0, got {position_pct}")

        # Try to extract <think> content
        think_content = self._extract_think_content(rollout)

        if think_content is not None:
            # Rollout has <think> tags - work with the content
            text_to_clip = think_content
        else:
            # No <think> tags - work with raw text
            text_to_clip = rollout

        # Calculate clip position by character count
        clip_position = int(len(text_to_clip) * position_pct)
        clipped_text = text_to_clip[:clip_position]

        # Add intervention text with formatting
        intervened_text = clipped_text + "\n\n" + intervention_text

        # Return with open <think> tag for continuation
        return f"<think>\n{intervened_text}\n"

    def _extract_think_content(self, text: str) -> Optional[str]:
        """
        Extract content from between <think> tags.

        Args:
            text: Text potentially containing <think> tags

        Returns:
            Content between tags, or None if no tags found
        """
        match = re.search(r'<think>(.*?)</think>', text, re.DOTALL)
        if match:
            return match.group(1)
        return None


class InterventionInserter:
    """
    Main class for applying interventions to rollouts.

    Uses a pluggable strategy pattern to allow different intervention approaches.
    """

    def __init__(self, strategy: Optional[InterventionStrategy] = None):
        """
        Initialize the intervention inserter.

        Args:
            strategy: The intervention strategy to use. Defaults to DirectInsertionStrategy.
        """
        self.strategy = strategy if strategy is not None else DirectInsertionStrategy()

    def clip_and_insert(
        self,
        rollout: str,
        intervention_text: str,
        position_pct: float = 0.5
    ) -> str:
        """
        Clip a rollout and insert intervention text.

        Args:
            rollout: The original rollout text
            intervention_text: Text to insert after clipping
            position_pct: Position to clip at (0.0 = start, 1.0 = end)

        Returns:
            Modified rollout ready for continuation

        Raises:
            ValueError: If position_pct is not in valid range

        Example:
            >>> inserter = InterventionInserter()
            >>> result = inserter.clip_and_insert(
            ...     rollout="<think>Let me think... maybe yes...</think>",
            ...     intervention_text="Wait, let me reconsider.",
            ...     position_pct=0.5
            ... )
        """
        return self.strategy.apply(rollout, intervention_text, position_pct)

    def set_strategy(self, strategy: InterventionStrategy):
        """
        Change the intervention strategy.

        Args:
            strategy: The new strategy to use
        """
        self.strategy = strategy


# Helper function for extracting think content (useful for analysis)
def extract_think_content(text: str) -> Optional[str]:
    """
    Extract content from between <think> tags.

    Args:
        text: Text potentially containing <think> tags

    Returns:
        Content between tags, or None if no tags found
    """
    match = re.search(r'<think>(.*?)</think>', text, re.DOTALL)
    if match:
        return match.group(1)
    return None
