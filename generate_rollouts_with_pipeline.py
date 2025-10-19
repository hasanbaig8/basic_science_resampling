#!/usr/bin/env python3
"""
Generate strategyqa_rollouts.json and strategyqa_rollouts_parsed.json using the pipeline.

This script replaces the old archive/generate_strategyqa_rollouts.py and
archive/parse_rollout_decisions.py scripts with the new pipeline architecture.
"""

import json
import argparse
import time
from pathlib import Path
from typing import List, Dict, Any

from pipeline.rollout_generator import RolloutGenerator
from pipeline.decision_parser import DecisionParser


def generate_rollouts(
    dataset_path: str,
    output_path: str,
    n_rollouts: int = 10,
    max_questions: int = None,
    start_index: int = 0,
    model_name: str = "Qwen/Qwen3-8b",
    vllm_url: str = "http://localhost:8000/v1/completions",
    max_tokens: int = 8192,
    temperature: float = 0.7
) -> List[Dict[str, Any]]:
    """
    Generate rollouts for StrategyQA questions.

    Args:
        dataset_path: Path to StrategyQA JSON file
        output_path: Where to save the rollouts
        n_rollouts: Number of rollouts per question
        max_questions: Maximum number of questions to process (None = all)
        start_index: Index of first question to process
        model_name: Name of the model on vLLM server
        vllm_url: URL of vLLM completions endpoint
        max_tokens: Maximum tokens to generate per rollout
        temperature: Sampling temperature for diversity

    Returns:
        List of dictionaries with structure:
        [{"question_id": int, "model_outputs": [str, ...]}, ...]
    """
    # Initialize pipeline components
    print(f"\nInitializing RolloutGenerator...")
    print(f"  Model: {model_name}")
    print(f"  vLLM URL: {vllm_url}")
    generator = RolloutGenerator(
        model_name=model_name,
        vllm_url=vllm_url,
        max_tokens=max_tokens,
        temperature=temperature
    )

    # Load dataset
    print(f"\nLoading dataset from {dataset_path}...")
    with open(dataset_path, 'r') as f:
        dataset = json.load(f)
    print(f"Loaded {len(dataset)} questions")

    # Determine which questions to process
    if max_questions is not None:
        end_index = min(start_index + max_questions, len(dataset))
        questions_to_process = dataset[start_index:end_index]
    else:
        questions_to_process = dataset[start_index:]

    print(f"\nProcessing {len(questions_to_process)} questions (starting from index {start_index})")
    print(f"Generating {n_rollouts} rollouts per question")
    print(f"Max tokens per rollout: {max_tokens}")
    print(f"Temperature: {temperature}")

    # Process each question
    results = []
    for i, item in enumerate(questions_to_process):
        question_id = item["question_id"]
        question = item["question"]
        answer = item["answer"]
        actual_index = start_index + i

        print(f"\n[{actual_index + 1}/{len(dataset)}] Processing question ID {question_id}:")
        print(f"  Q: {question}")
        print(f"  A: {answer}")

        try:
            # Generate rollouts using pipeline
            print(f"  Generating {n_rollouts} rollouts...", end=" ", flush=True)
            rollouts = generator.generate_from_question(question, n=n_rollouts)
            print(f"✓ Done ({len(rollouts)} rollouts)")

            # Store result with only question_id and model_outputs
            result = {
                "question_id": question_id,
                "model_outputs": rollouts
            }
            results.append(result)

            # Save incrementally (in case of interruption)
            save_results(results, output_path)

            # Small delay to avoid overwhelming server
            time.sleep(0.5)

        except Exception as e:
            print(f"✗ FAILED: {e}")
            continue

    return results


def save_results(results: List[Dict[str, Any]], output_path: str):
    """Save results to JSON file."""
    # Ensure directory exists
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"  Saved to: {output_path}")


def parse_rollouts(
    rollouts_path: str,
    output_path: str
) -> List[Dict[str, Any]]:
    """
    Parse decisions from rollouts file.

    Args:
        rollouts_path: Path to strategyqa_rollouts.json
        output_path: Where to save parsed decisions

    Returns:
        List of dictionaries with structure:
        [{"question_id": int, "decisions": [bool/None, ...]}, ...]
    """
    # Initialize parser
    print(f"\nInitializing DecisionParser...")
    parser = DecisionParser()

    # Load rollouts
    print(f"Loading rollouts from {rollouts_path}...")
    with open(rollouts_path, 'r') as f:
        rollouts_data = json.load(f)
    print(f"Loaded {len(rollouts_data)} questions")

    # Parse decisions for each question
    parsed_results = []
    for i, item in enumerate(rollouts_data):
        question_id = item["question_id"]
        model_outputs = item["model_outputs"]

        print(f"[{i + 1}/{len(rollouts_data)}] Parsing question ID {question_id}...", end=" ")

        # Parse decisions using pipeline
        decisions = parser.parse_multiple(model_outputs)

        # Store result
        parsed_result = {
            "question_id": question_id,
            "decisions": decisions
        }
        parsed_results.append(parsed_result)

        print(f"✓ {len(decisions)} decisions parsed")

    # Save parsed results
    save_results(parsed_results, output_path)

    return parsed_results


def print_sample_rollout(
    results: List[Dict[str, Any]],
    dataset_path: str,
    question_idx: int = 0,
    rollout_idx: int = 0
):
    """Print a sample rollout for inspection."""
    if not results or question_idx >= len(results):
        print("No results to display")
        return

    result = results[question_idx]
    if rollout_idx >= len(result["model_outputs"]):
        print(f"Rollout {rollout_idx} not found")
        return

    # Load the full dataset to get question and answer for display
    with open(dataset_path, 'r') as f:
        dataset = json.load(f)

    question_id = result["question_id"]
    question_data = next((item for item in dataset if item["question_id"] == question_id), None)

    if not question_data:
        print(f"Question ID {question_id} not found in dataset")
        return

    print(f"\n{'='*80}")
    print(f"SAMPLE ROLLOUT")
    print(f"{'='*80}")
    print(f"Question ID: {question_id}")
    print(f"Question: {question_data['question']}")
    print(f"Answer: {question_data['answer']}")
    print(f"\nRollout {rollout_idx + 1}:")
    print(f"{'-'*80}")
    print(result["model_outputs"][rollout_idx])
    print(f"{'-'*80}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate rollouts and parsed decisions for StrategyQA using pipeline"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="data/strategyqa_data.json",
        help="Path to StrategyQA dataset JSON file"
    )
    parser.add_argument(
        "--rollouts-output",
        type=str,
        default="data/strategyqa_rollouts.json",
        help="Output path for rollouts JSON file"
    )
    parser.add_argument(
        "--parsed-output",
        type=str,
        default="data/strategyqa_rollouts_parsed.json",
        help="Output path for parsed decisions JSON file"
    )
    parser.add_argument(
        "--n-rollouts",
        type=int,
        default=100,
        help="Number of rollouts per question"
    )
    parser.add_argument(
        "--max-questions",
        type=int,
        default=None,
        help="Maximum number of questions to process (default: all)"
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Index of first question to process"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=8192,
        help="Maximum tokens to generate per rollout"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3-8b",
        help="Model name on vLLM server"
    )
    parser.add_argument(
        "--url",
        type=str,
        default="http://localhost:8000/v1/completions",
        help="vLLM completions endpoint URL"
    )
    parser.add_argument(
        "--skip-generation",
        action="store_true",
        help="Skip rollout generation, only parse existing rollouts"
    )

    args = parser.parse_args()

    print(f"\n{'='*80}")
    print("STRATEGYQA ROLLOUT GENERATOR (PIPELINE VERSION)")
    print(f"{'='*80}")

    # Step 1: Generate rollouts (unless skipped)
    if not args.skip_generation:
        print("\n### STEP 1: GENERATING ROLLOUTS ###")
        results = generate_rollouts(
            dataset_path=args.dataset,
            output_path=args.rollouts_output,
            n_rollouts=args.n_rollouts,
            max_questions=args.max_questions,
            start_index=args.start_index,
            model_name=args.model,
            vllm_url=args.url,
            max_tokens=args.max_tokens,
            temperature=args.temperature
        )

        print(f"\n{'='*80}")
        print("ROLLOUT GENERATION COMPLETE!")
        print(f"{'='*80}")
        print(f"Processed {len(results)} questions")
        print(f"Saved to: {args.rollouts_output}")

        # Show a sample rollout
        if results:
            print_sample_rollout(results, dataset_path=args.dataset, question_idx=0, rollout_idx=0)
    else:
        print("\n### SKIPPING ROLLOUT GENERATION ###")

    # Step 2: Parse decisions
    print(f"\n{'='*80}")
    print("### STEP 2: PARSING DECISIONS ###")
    print(f"{'='*80}")

    parsed_results = parse_rollouts(
        rollouts_path=args.rollouts_output,
        output_path=args.parsed_output
    )

    print(f"\n{'='*80}")
    print("PARSING COMPLETE!")
    print(f"{'='*80}")
    print(f"Parsed {len(parsed_results)} questions")
    print(f"Saved to: {args.parsed_output}")

    # Print summary statistics
    if parsed_results:
        print(f"\n{'='*80}")
        print("SUMMARY STATISTICS")
        print(f"{'='*80}")

        total_decisions = 0
        total_true = 0
        total_false = 0
        total_none = 0

        for result in parsed_results:
            decisions = result["decisions"]
            total_decisions += len(decisions)
            total_true += sum(1 for d in decisions if d is True)
            total_false += sum(1 for d in decisions if d is False)
            total_none += sum(1 for d in decisions if d is None)

        print(f"Total decisions: {total_decisions}")
        print(f"  True:  {total_true} ({100*total_true/total_decisions:.1f}%)")
        print(f"  False: {total_false} ({100*total_false/total_decisions:.1f}%)")
        print(f"  None:  {total_none} ({100*total_none/total_decisions:.1f}%)")


if __name__ == "__main__":
    main()
