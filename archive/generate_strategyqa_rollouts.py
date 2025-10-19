#!/usr/bin/env python3
"""
Generate rollouts for StrategyQA questions using vLLM server.

Generates multiple completions (rollouts) for each question, capturing the full
model output including any <think> tags used for chain-of-thought reasoning.
"""

import requests
import json
import argparse
from typing import List, Dict, Any
import time
from pathlib import Path
from transformers import AutoTokenizer


class StrategyQARolloutGenerator:
    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-8b",
        vllm_url: str = "http://localhost:8000/v1/completions",
        max_tokens: int = 8192,
        temperature: float = 0.7
    ):
        """
        Initialize the rollout generator.

        Args:
            model_name: Name of the model on vLLM server
            vllm_url: URL of vLLM completions endpoint
            max_tokens: Maximum tokens to generate per rollout
            temperature: Sampling temperature for diversity
        """
        self.model_name = model_name
        self.vllm_url = vllm_url
        self.max_tokens = max_tokens
        self.temperature = temperature

        # Load tokenizer for chat template
        print(f"Loading tokenizer for {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        print("Tokenizer loaded!")

    def format_prompt(self, question: str) -> str:
        """
        Format question using tokenizer's chat template with JSON output instruction.

        Args:
            question: The question to format

        Returns:
            Formatted prompt string
        """
        user_message = f"""Answer the following yes/no question.

Question: {question}

Provide your final answer as a JSON object: {{"decision": true}} or {{"decision": false}}"""

        messages = [
            {"role": "user", "content": user_message}
        ]

        # Use the tokenizer's chat template with add_generation_prompt=True
        formatted_prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        return formatted_prompt

    def generate_rollouts(
        self,
        question: str,
        n_rollouts: int = 10
    ) -> List[str]:
        """
        Generate multiple rollouts for a single question.

        Args:
            question: The question to answer
            n_rollouts: Number of rollouts to generate

        Returns:
            List of generated text completions
        """
        formatted_prompt = self.format_prompt(question)

        payload = {
            "model": self.model_name,
            "prompt": formatted_prompt,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "stream": False,
            "n": n_rollouts
        }

        headers = {"Content-Type": "application/json"}

        try:
            print(f"  Generating {n_rollouts} rollouts...", end=" ", flush=True)
            response = requests.post(self.vllm_url, json=payload, headers=headers)
            response.raise_for_status()
            result = response.json()

            # Extract all completion texts
            rollouts = [choice["text"] for choice in result["choices"]]
            print(f"✓ Done ({len(rollouts)} rollouts)")
            return rollouts

        except Exception as e:
            print(f"✗ FAILED: {e}")
            return []

    def process_dataset(
        self,
        dataset_path: str,
        output_path: str,
        n_rollouts: int = 10,
        max_questions: int = None,
        start_index: int = 0
    ) -> List[Dict[str, Any]]:
        """
        Process questions from dataset and generate rollouts.

        Args:
            dataset_path: Path to StrategyQA JSON file
            output_path: Where to save the rollouts
            n_rollouts: Number of rollouts per question
            max_questions: Maximum number of questions to process (None = all)
            start_index: Index of first question to process

        Returns:
            List of processed question-rollout dictionaries
        """
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

        print(f"Processing {len(questions_to_process)} questions (starting from index {start_index})")
        print(f"Generating {n_rollouts} rollouts per question")
        print(f"Max tokens per rollout: {self.max_tokens}")
        print(f"Temperature: {self.temperature}")

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

            # Generate rollouts
            rollouts = self.generate_rollouts(question, n_rollouts=n_rollouts)

            if not rollouts:
                print(f"  Warning: No rollouts generated for question_id {question_id}")
                continue

            # Store result with only question_id and model_outputs
            result = {
                "question_id": question_id,
                "model_outputs": rollouts
            }
            results.append(result)

            # Save incrementally (in case of interruption)
            self._save_results(results, output_path)

            # Small delay to avoid overwhelming server
            time.sleep(0.5)

        return results

    def _save_results(self, results: List[Dict[str, Any]], output_path: str):
        """Save results to JSON file."""
        # Ensure directory exists
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

    def print_sample_rollout(self, results: List[Dict[str, Any]], dataset_path: str, question_idx: int = 0, rollout_idx: int = 0):
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
        description="Generate rollouts for StrategyQA questions using vLLM"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="data/strategyqa_data.json",
        help="Path to StrategyQA dataset JSON file"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/strategyqa_rollouts.json",
        help="Output path for rollouts JSON file"
    )
    parser.add_argument(
        "--n-rollouts",
        type=int,
        default=10,
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

    args = parser.parse_args()

    # Create generator
    print(f"\n{'='*80}")
    print("STRATEGYQA ROLLOUT GENERATOR")
    print(f"{'='*80}")
    print(f"Model: {args.model}")
    print(f"vLLM URL: {args.url}")

    generator = StrategyQARolloutGenerator(
        model_name=args.model,
        vllm_url=args.url,
        max_tokens=args.max_tokens,
        temperature=args.temperature
    )

    # Process dataset
    results = generator.process_dataset(
        dataset_path=args.dataset,
        output_path=args.output,
        n_rollouts=args.n_rollouts,
        max_questions=args.max_questions,
        start_index=args.start_index
    )

    # Print summary
    print(f"\n{'='*80}")
    print("GENERATION COMPLETE!")
    print(f"{'='*80}")
    print(f"Processed {len(results)} questions")
    print(f"Saved to: {args.output}")

    # Show a sample rollout
    if results:
        generator.print_sample_rollout(results, dataset_path=args.dataset, question_idx=0, rollout_idx=0)


if __name__ == "__main__":
    main()
