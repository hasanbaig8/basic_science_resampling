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
    (e.g., paraphrasing, contextual insertion, semantic boundary insertion, etc.)

    Strategy-specific parameters (like position, patterns, etc.) should be
    configured in the strategy's __init__() method, not passed to apply().
    """

    @abstractmethod
    def apply(self, rollout: str, intervention_text: str, prompt: Optional[str] = None) -> str:
        """
        Apply the intervention to a rollout.

        Args:
            rollout: The original rollout text (may include <think> tags)
            intervention_text: Text to insert
            prompt: Optional formatted prompt (useful for context-aware strategies)

        Returns:
            Modified text ready for continuation (with open <think> tag if applicable)

        Note:
            Strategy-specific parameters should be set during initialization,
            not passed to this method. This keeps the interface consistent
            across all strategies.
        """
        pass


class DirectInsertionStrategy(InterventionStrategy):
    """
    Simple strategy: clip at position and insert text directly.

    This is the baseline intervention approach.

    Args:
        position_pct: Position to clip at (0.0-1.0). Default 0.5 (halfway).
    """

    def __init__(self, position_pct: float = 0.5):
        """
        Initialize DirectInsertionStrategy.

        Args:
            position_pct: Position to clip at (0.0-1.0, where 0.5=halfway)

        Raises:
            ValueError: If position_pct is not in valid range
        """
        if not 0.0 <= position_pct <= 1.0:
            raise ValueError(f"position_pct must be between 0.0 and 1.0, got {position_pct}")
        self.position_pct = position_pct

    def apply(self, rollout: str, intervention_text: str, prompt: Optional[str] = None) -> str:
        """
        Clip rollout and insert intervention text.

        If rollout has <think> tags, extracts content and preserves structure.
        Otherwise, clips the raw text.

        Args:
            rollout: The original rollout text
            intervention_text: Text to insert after clipping
            prompt: Optional formatted prompt (not used in DirectInsertionStrategy)

        Returns:
            Clipped text with intervention inserted, ending with open <think> tag
        """
        # Try to extract <think> content
        think_content = self._extract_think_content(rollout)

        if think_content is not None:
            # Rollout has <think> tags - work with the content
            text_to_clip = think_content
        else:
            # No <think> tags - work with raw text
            text_to_clip = rollout

        # Calculate clip position by character count
        clip_position = int(len(text_to_clip) * self.position_pct)
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

    def apply(
        self,
        rollout: str,
        intervention_text: str,
        prompt: Optional[str] = None
    ) -> str:
        """
        Clip a rollout and insert intervention text using the configured strategy.

        Args:
            rollout: The original rollout text
            intervention_text: Text to insert after clipping
            prompt: Optional formatted prompt (for context-aware strategies)

        Returns:
            Modified rollout ready for continuation

        Note:
            Strategy-specific parameters (like position) should be configured
            when creating the strategy instance, not passed here.

        Example:
            >>> strategy = DirectInsertionStrategy(position_pct=0.5)
            >>> inserter = InterventionInserter(strategy=strategy)
            >>> result = inserter.apply(
            ...     rollout="<think>Let me think... maybe yes...</think>",
            ...     intervention_text="Wait, let me reconsider."
            ... )
        """
        return self.strategy.apply(rollout, intervention_text, prompt)

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
