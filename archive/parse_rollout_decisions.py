#!/usr/bin/env python3
"""
Parse decision values from StrategyQA rollouts.

Extracts the JSON {"decision": true/false} from each rollout and saves
a simplified format with just question_id and a list of boolean decisions.
"""

import json
import re
import argparse
from typing import Optional


def extract_decision(rollout_text: str) -> Optional[bool]:
    """
    Extract the decision boolean from a rollout text.

    Looks for JSON patterns like {"decision": true} or {"decision": false}

    Args:
        rollout_text: The full text of a rollout

    Returns:
        True, False, or None if no decision found
    """
    # Try to find {"decision": true} or {"decision": false}
    # Handle various spacing and quote styles
    patterns = [
        r'\{\s*"decision"\s*:\s*true\s*\}',
        r'\{\s*"decision"\s*:\s*false\s*\}',
        r"\{\s*'decision'\s*:\s*true\s*\}",
        r"\{\s*'decision'\s*:\s*false\s*\}",
    ]

    for pattern in patterns:
        match = re.search(pattern, rollout_text, re.IGNORECASE)
        if match:
            # Check if it contains 'true'
            if 'true' in match.group().lower():
                return True
            else:
                return False

    return None


def parse_rollouts(input_path: str, output_path: str, verbose: bool = True):
    """
    Parse rollouts file and extract decisions.

    Args:
        input_path: Path to rollouts JSON file
        output_path: Path to save parsed results
        verbose: Whether to print progress
    """
    if verbose:
        print(f"Loading rollouts from {input_path}...")

    with open(input_path, 'r', encoding='utf-8') as f:
        rollouts_data = json.load(f)

    if verbose:
        print(f"Loaded {len(rollouts_data)} questions")

    parsed_results = []
    total_rollouts = 0
    failed_parses = 0

    for item in rollouts_data:
        question_id = item["question_id"]
        model_outputs = item["model_outputs"]

        # Extract decision from each rollout
        decisions = []
        successful_parse = []
        for i, rollout in enumerate(model_outputs):
            decision = extract_decision(rollout)

            if decision is None:
                if verbose:
                    print(f"  Warning: Could not parse decision for question_id {question_id}, rollout {i}")
                failed_parses += 1
                successful_parse.append(False)
            else:
                successful_parse.append(True)

            decisions.append(decision)
            total_rollouts += 1

        parsed_results.append({
            "question_id": question_id,
            "decisions": decisions,
            "successful_parse": successful_parse
        })

    # Save parsed results
    if verbose:
        print(f"\nSaving parsed results to {output_path}...")

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(parsed_results, f, indent=2, ensure_ascii=False)

    # Print summary
    if verbose:
        print(f"\n{'='*80}")
        print("PARSING COMPLETE!")
        print(f"{'='*80}")
        print(f"Total questions: {len(parsed_results)}")
        print(f"Total rollouts: {total_rollouts}")
        print(f"Successfully parsed: {total_rollouts - failed_parses}")
        print(f"Failed to parse: {failed_parses}")
        print(f"Success rate: {((total_rollouts - failed_parses) / total_rollouts * 100):.2f}%")
        print(f"Saved to: {output_path}")

        # Show a sample
        if parsed_results:
            sample = parsed_results[0]
            print(f"\nSample entry:")
            print(f"  Question ID: {sample['question_id']}")
            print(f"  Decisions: {sample['decisions'][:5]}{'...' if len(sample['decisions']) > 5 else ''}")
            print(f"  Successful parse: {sample['successful_parse'][:5]}{'...' if len(sample['successful_parse']) > 5 else ''}")
            print(f"  Total decisions: {len(sample['decisions'])}")


def main():
    parser = argparse.ArgumentParser(
        description="Parse decisions from StrategyQA rollouts"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="data/strategyqa_rollouts.json",
        help="Path to rollouts JSON file"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/strategyqa_rollouts_parsed.json",
        help="Path to save parsed results"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress output"
    )

    args = parser.parse_args()

    parse_rollouts(
        input_path=args.input,
        output_path=args.output,
        verbose=not args.quiet
    )


if __name__ == "__main__":
    main()
