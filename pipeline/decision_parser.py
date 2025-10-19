#!/usr/bin/env python3
"""
DecisionParser - Extract boolean decisions from rollout text.

Parses JSON-formatted decisions like {"decision": true} or {"decision": false}
from model-generated text.
"""

import re
from typing import Optional, List, Dict


class DecisionParser:
    """
    Parse boolean decisions from rollout text.

    Looks for JSON patterns like {"decision": true} or {"decision": false}
    and extracts the boolean value.
    """

    def __init__(self):
        """Initialize the decision parser with regex patterns."""
        # Patterns to match various formats of decision JSON
        self.patterns = [
            r'\{\s*"decision"\s*:\s*true\s*\}',
            r'\{\s*"decision"\s*:\s*false\s*\}',
            r"\{\s*'decision'\s*:\s*true\s*\}",
            r"\{\s*'decision'\s*:\s*false\s*\}",
        ]

    def parse_decision(self, rollout_text: str) -> Optional[bool]:
        """
        Extract the decision boolean from a rollout text.

        Args:
            rollout_text: The full text of a rollout

        Returns:
            True, False, or None if no decision found

        Example:
            >>> parser = DecisionParser()
            >>> parser.parse_decision('I think the answer is {"decision": true}')
            True
            >>> parser.parse_decision('No clear decision here')
            None
        """
        for pattern in self.patterns:
            match = re.search(pattern, rollout_text, re.IGNORECASE)
            if match:
                # Check if it contains 'true'
                return 'true' in match.group().lower()

        return None

    def parse_multiple(self, rollout_texts: List[str]) -> List[Optional[bool]]:
        """
        Parse decisions from multiple rollouts.

        Args:
            rollout_texts: List of rollout text strings

        Returns:
            List of boolean decisions (True/False/None for each rollout)

        Example:
            >>> parser = DecisionParser()
            >>> texts = ['{"decision": true}', '{"decision": false}', 'unclear']
            >>> parser.parse_multiple(texts)
            [True, False, None]
        """
        return [self.parse_decision(text) for text in rollout_texts]

    def parse_with_success_indicator(
        self,
        rollout_texts: List[str]
    ) -> Dict[str, List]:
        """
        Parse decisions and indicate which parses succeeded.

        Args:
            rollout_texts: List of rollout text strings

        Returns:
            Dictionary with:
                - 'decisions': List of Optional[bool]
                - 'successful_parse': List of bool indicating if parse succeeded

        Example:
            >>> parser = DecisionParser()
            >>> result = parser.parse_with_success_indicator(['{"decision": true}', 'unclear'])
            >>> result['decisions']
            [True, None]
            >>> result['successful_parse']
            [True, False]
        """
        decisions = []
        successful_parse = []

        for text in rollout_texts:
            decision = self.parse_decision(text)
            decisions.append(decision)
            successful_parse.append(decision is not None)

        return {
            'decisions': decisions,
            'successful_parse': successful_parse
        }

    def get_parse_success_rate(self, rollout_texts: List[str]) -> float:
        """
        Calculate the percentage of rollouts that were successfully parsed.

        Args:
            rollout_texts: List of rollout text strings

        Returns:
            Success rate as a float between 0.0 and 1.0

        Example:
            >>> parser = DecisionParser()
            >>> texts = ['{"decision": true}', '{"decision": false}', 'unclear']
            >>> parser.get_parse_success_rate(texts)
            0.6666666666666666
        """
        if not rollout_texts:
            return 0.0

        decisions = self.parse_multiple(rollout_texts)
        successful = sum(1 for d in decisions if d is not None)
        return successful / len(rollout_texts)
