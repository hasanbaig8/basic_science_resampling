#!/usr/bin/env python3
"""
Run interventions on existing CoT rollouts.

Loads rollouts, applies interventions at specified positions, continues generation,
and saves results to a separate interventions file.
"""

import requests
import json
import argparse
from typing import List, Dict, Any, Optional
import time
from pathlib import Path
from transformers import AutoTokenizer

from interventions import (
    CoTIntervention,
    DirectInsertionIntervention,
    ParaphrasingIntervention,
    extract_think_content
)


class InterventionRunner:
    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-8b",
        vllm_url: str = "http://localhost:8000/v1/completions",
        max_tokens: int = 8192,
        temperature: float = 0.7
    ):
        """
        Initialize the intervention runner.

        Args:
            model_name: Name of the model on vLLM server
            vllm_url: URL of vLLM completions endpoint
            max_tokens: Maximum tokens to generate per continuation
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

    def continue_generation(
        self,
        original_prompt: str,
        intervened_think: str
    ) -> str:
        """
        Continue generation from an intervened CoT.

        Args:
            original_prompt: The original user prompt (formatted with chat template)
            intervened_think: The clipped + intervened <think> content

        Returns:
            Generated continuation text
        """
        # Combine original prompt with intervened think to form the new prompt
        # The intervened_think should end with an open <think> tag for continuation
        full_prompt = original_prompt + intervened_think

        payload = {
            "model": self.model_name,
            "prompt": full_prompt,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "stream": False,
            "n": 1
        }

        headers = {"Content-Type": "application/json"}

        try:
            response = requests.post(self.vllm_url, json=payload, headers=headers)
            response.raise_for_status()
            result = response.json()

            # Extract the continuation text
            continuation = result["choices"][0]["text"]
            return continuation

        except Exception as e:
            print(f"  Error generating continuation: {e}")
            return ""

    def format_question_prompt(self, question: str) -> str:
        """
        Format question using tokenizer's chat template.

        Args:
            question: The question to format

        Returns:
            Formatted prompt string
        """
        user_message = f"""Answer the following yes/no question.

Question: {question}

Provide your final answer as a JSON object: {{"decision": true}} or {{"decision": false}}"""

        messages = [{"role": "user", "content": user_message}]

        # Use the tokenizer's chat template with add_generation_prompt=True
        formatted_prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        return formatted_prompt

    def run_interventions(
        self,
        rollouts_path: str,
        questions_path: str,
        output_path: str,
        intervention: CoTIntervention,
        intervention_type: str,
        intervention_text: str,
        clip_position_pct: float,
        steerable_question_ids_path: Optional[str] = None,
        max_questions: Optional[int] = None,
        rollout_indices: Optional[List[int]] = None
    ) -> List[Dict[str, Any]]:
        """
        Run interventions on rollouts.

        Args:
            rollouts_path: Path to rollouts JSON file
            questions_path: Path to questions JSON file
            output_path: Where to save intervention results
            intervention: The intervention object to apply
            intervention_type: String describing intervention type
            intervention_text: The intervention text used
            clip_position_pct: Position where clipping occurred
            steerable_question_ids_path: Optional path to steerable question IDs JSON file
            max_questions: Maximum number of questions to process
            rollout_indices: Which rollout indices to intervene on (default: [0])

        Returns:
            List of intervention result dictionaries
        """
        # Load rollouts and questions
        print(f"\nLoading rollouts from {rollouts_path}...")
        with open(rollouts_path, 'r') as f:
            rollouts_data = json.load(f)

        print(f"Loading questions from {questions_path}...")
        with open(questions_path, 'r') as f:
            questions_data = json.load(f)

        # Load steerable question IDs if provided
        steerable_question_ids = None
        if steerable_question_ids_path:
            print(f"Loading steerable question IDs from {steerable_question_ids_path}...")
            with open(steerable_question_ids_path, 'r') as f:
                steerable_data = json.load(f)
                steerable_question_ids = set(steerable_data["question_ids"])
            print(f"Loaded {len(steerable_question_ids)} steerable question IDs")

        # Create question lookup
        questions_lookup = {q["question_id"]: q for q in questions_data}

        # Filter rollouts to only steerable questions if specified
        if steerable_question_ids is not None:
            rollouts_data = [r for r in rollouts_data if r["question_id"] in steerable_question_ids]
            print(f"Filtered to {len(rollouts_data)} steerable questions")

        # Determine which rollouts to process
        if max_questions is not None:
            rollouts_data = rollouts_data[:max_questions]

        if rollout_indices is None:
            rollout_indices = [0]

        print(f"Processing {len(rollouts_data)} questions")
        print(f"Intervention type: {intervention_type}")
        print(f"Clip position: {clip_position_pct * 100:.1f}%")
        print(f"Rollout indices: {rollout_indices}")

        results = []

        for question_data in rollouts_data:
            question_id = question_data["question_id"]
            model_outputs = question_data["model_outputs"]

            # Get question details
            question_info = questions_lookup.get(question_id)
            if not question_info:
                print(f"  Warning: Question ID {question_id} not found in questions file")
                continue

            question_text = question_info["question"]
            answer = question_info["answer"]

            print(f"\n[Question {question_id}] {question_text}")
            print(f"  Answer: {answer}")

            # Process specified rollout indices
            for rollout_idx in rollout_indices:
                if rollout_idx >= len(model_outputs):
                    print(f"  Warning: Rollout index {rollout_idx} out of range")
                    continue

                original_output = model_outputs[rollout_idx]

                print(f"  Processing rollout {rollout_idx}...", end=" ", flush=True)

                try:
                    # Apply intervention
                    intervened_think = intervention.apply(original_output)

                    # Get the original formatted prompt
                    original_prompt = self.format_question_prompt(question_text)

                    # Continue generation
                    continued_output = self.continue_generation(
                        original_prompt,
                        intervened_think
                    )

                    if not continued_output:
                        print("✗ Failed")
                        continue

                    # Combine intervened think with continuation
                    full_output = intervened_think + continued_output

                    # Try to parse decision from the full output
                    decision = self._extract_decision(full_output)

                    result = {
                        "question_id": question_id,
                        "rollout_index": rollout_idx,
                        "intervention_type": intervention_type,
                        "clip_position_pct": clip_position_pct,
                        "intervention_text": intervention_text,
                        "original_output": original_output,
                        "intervened_think": intervened_think,
                        "continued_output": continued_output,
                        "full_output": full_output,
                        "decision": decision,
                        "successful_parse": decision is not None
                    }

                    results.append(result)
                    print(f"✓ Done (decision: {decision})")

                    # Save incrementally
                    self._save_results(results, output_path)

                except Exception as e:
                    print(f"✗ Error: {e}")
                    continue

                # Small delay
                time.sleep(0.5)

        return results

    def _extract_decision(self, text: str) -> Optional[bool]:
        """Extract decision from output text."""
        import re

        patterns = [
            r'\{\s*"decision"\s*:\s*true\s*\}',
            r'\{\s*"decision"\s*:\s*false\s*\}',
        ]

        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return 'true' in match.group().lower()

        return None

    def _save_results(self, results: List[Dict[str, Any]], output_path: str):
        """Save results to JSON file."""
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser(
        description="Run interventions on StrategyQA rollouts"
    )
    parser.add_argument(
        "--rollouts",
        type=str,
        default="data/strategyqa_rollouts.json",
        help="Path to rollouts JSON file"
    )
    parser.add_argument(
        "--questions",
        type=str,
        default="data/strategyqa_data.json",
        help="Path to questions JSON file"
    )
    parser.add_argument(
        "--steerable-question-ids",
        type=str,
        default=None,
        help="Path to steerable question IDs JSON file (optional)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/strategyqa_interventions.json",
        help="Output path for intervention results"
    )
    parser.add_argument(
        "--intervention-type",
        type=str,
        default="direct_insertion",
        choices=["direct_insertion", "paraphrasing"],
        help="Type of intervention to apply"
    )
    parser.add_argument(
        "--intervention-text",
        type=str,
        required=True,
        help="Text to insert as intervention"
    )
    parser.add_argument(
        "--clip-position",
        type=float,
        default=0.5,
        help="Position to clip CoT (0.0-1.0, e.g., 0.25=early, 0.5=mid, 0.75=late)"
    )
    parser.add_argument(
        "--max-questions",
        type=int,
        default=None,
        help="Maximum number of questions to process"
    )
    parser.add_argument(
        "--rollout-indices",
        type=int,
        nargs="+",
        default=[0],
        help="Which rollout indices to intervene on (e.g., 0 1 2)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3-8b",
        help="Model name"
    )
    parser.add_argument(
        "--url",
        type=str,
        default="http://localhost:8000/v1/completions",
        help="vLLM URL"
    )

    args = parser.parse_args()

    # Create intervention object
    if args.intervention_type == "direct_insertion":
        intervention = DirectInsertionIntervention(
            intervention_text=args.intervention_text,
            position_pct=args.clip_position
        )
    elif args.intervention_type == "paraphrasing":
        intervention = ParaphrasingIntervention(
            intervention_text=args.intervention_text,
            position_pct=args.clip_position,
            paraphrasing_fn=None  # You can provide custom paraphrasing function
        )
    else:
        raise ValueError(f"Unknown intervention type: {args.intervention_type}")

    # Create runner
    print(f"\n{'='*80}")
    print("INTERVENTION RUNNER")
    print(f"{'='*80}")
    print(f"Model: {args.model}")
    print(f"vLLM URL: {args.url}")

    runner = InterventionRunner(
        model_name=args.model,
        vllm_url=args.url
    )

    # Run interventions
    results = runner.run_interventions(
        rollouts_path=args.rollouts,
        questions_path=args.questions,
        output_path=args.output,
        intervention=intervention,
        intervention_type=args.intervention_type,
        intervention_text=args.intervention_text,
        clip_position_pct=args.clip_position,
        steerable_question_ids_path=args.steerable_question_ids,
        max_questions=args.max_questions,
        rollout_indices=args.rollout_indices
    )

    # Print summary
    print(f"\n{'='*80}")
    print("INTERVENTION COMPLETE!")
    print(f"{'='*80}")
    print(f"Processed {len(results)} interventions")
    print(f"Saved to: {args.output}")

    # Show success rate
    successful = sum(1 for r in results if r["successful_parse"])
    print(f"Successfully parsed decisions: {successful}/{len(results)} ({successful/len(results)*100:.1f}%)")


if __name__ == "__main__":
    main()
