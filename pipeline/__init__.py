"""
Pipeline for CoT Intervention Experiments

This package provides a clean interface for running intervention experiments
on chain-of-thought reasoning in LLMs.

Main components:
- RolloutGenerator: Generate completions using vLLM
- InterventionInserter: Clip and insert intervention text
- DecisionParser: Parse boolean decisions from rollouts
- analysis_utils: Statistical analysis functions

Example usage:
    >>> from pipeline import RolloutGenerator, InterventionInserter, DecisionParser
    >>> from pipeline.analysis_utils import compute_statistics
    >>>
    >>> # Generate initial rollouts
    >>> generator = RolloutGenerator()
    >>> rollouts = generator.generate_from_question("Is the sky blue?", n=10)
    >>>
    >>> # Apply intervention
    >>> inserter = InterventionInserter()
    >>> intervened = inserter.apply(
    ...     rollouts[0],
    ...     "Wait, let me reconsider.",
    ...     position_pct=0.5
    ... )
    >>>
    >>> # Continue generation
    >>> formatted_prompt = generator.format_question_prompt("Is the sky blue?")
    >>> continuations = generator.continue_generation(formatted_prompt, intervened)
    >>>
    >>> # Parse decisions
    >>> parser = DecisionParser()
    >>> decisions = parser.parse_multiple(continuations)
    >>> stats = compute_statistics(decisions)
"""

from .rollout_generator import RolloutGenerator
from .intervention_inserter import (
    InterventionInserter,
    InterventionStrategy,
    DirectInsertionStrategy,
    extract_think_content
)
from .voice_in_head_strategy import VoiceInHeadStrategy
from .decision_parser import DecisionParser

__all__ = [
    'RolloutGenerator',
    'InterventionInserter',
    'InterventionStrategy',
    'DirectInsertionStrategy',
    'VoiceInHeadStrategy',
    'DecisionParser',
    'extract_think_content'
]

__version__ = '1.0.0'
