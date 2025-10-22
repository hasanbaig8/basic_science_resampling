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

    def __init__(self, require_think_tag: bool = False):
        """
        Initialize the decision parser.
        
        Args:
            require_think_tag: If True, only look for decisions after </think> tag
        """
        self.require_think_tag = require_think_tag
        
        # Single flexible pattern to match JSON-like objects with decision key
        if require_think_tag:
            # Look for decisions after </think> tag
            self.pattern = r'</think>.*?\{\s*["\']?decision["\']?\s*:\s*(true|false)\s*\}'
        else:
            # Look for decisions anywhere in the text
            self.pattern = r'\{\s*["\']?decision["\']?\s*:\s*(true|false)\s*\}'

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
        import json
        
        # First try to find JSON-like decision patterns
        match = re.search(self.pattern, rollout_text, re.IGNORECASE | re.DOTALL)
        if match:
            decision_value = match.group(1).lower()
            return decision_value == 'true'
        
        # Fallback: try to find and parse complete JSON objects
        json_pattern = r'\{[^{}]*"decision"[^{}]*\}'
        json_matches = re.findall(json_pattern, rollout_text, re.IGNORECASE)
        
        for json_str in json_matches:
            try:
                # Clean up the JSON string and attempt to parse
                cleaned_json = re.sub(r"'", '"', json_str)  # Replace single quotes
                data = json.loads(cleaned_json)
                if 'decision' in data:
                    decision = data['decision']
                    if isinstance(decision, bool):
                        return decision
                    elif isinstance(decision, str):
                        return decision.lower() == 'true'
            except (json.JSONDecodeError, KeyError, TypeError):
                continue
        
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