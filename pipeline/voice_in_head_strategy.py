#!/usr/bin/env python3
"""
VoiceInHeadStrategy - Intervention strategy that inserts at a random early position.

This strategy simulates a "voice in head" that interrupts early in the reasoning process
by inserting at a random point in the first 15-35% of the text.
"""

import random
from typing import Optional
import requests
from .intervention_inserter import InterventionStrategy
from .rollout_generator import RolloutGenerator
from .intervention_grader import InterventionGrader
from transformers import AutoTokenizer
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os

class VoiceInHeadStrategy(InterventionStrategy):
    """
    Strategy: Insert at a random point in the first 15-35% of the text.

    Simulates a "voice in head" that interrupts early in the reasoning process.
    """

    def __init__(self):
        """Initialize VoiceInHeadStrategy."""
        self.generator = RolloutGenerator(max_tokens = 100)
        self.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8b")

        # Initialize LLM-based grader for scoring interventions
        self.grader = InterventionGrader()
        print(f"Initialized InterventionGrader using vLLM")

        # vLLM URL for logprobs
        self.vllm_url = "http://localhost:8000/v1/completions"
        self.n_interventions = 10

    def apply(self, rollout: str, intervention_text: str, prompt: Optional[str] = None, plot: bool = False) -> str:
        """
        Insert intervention at a random point in the first 15-35% of the rollout.

        Args:
            rollout: The original rollout text
            intervention_text: Text to insert
            prompt: Optional formatted prompt (not used in VoiceInHeadStrategy)

        Returns:
            Clipped text with intervention inserted, ending with open <think> tag
        """
        # Try to extract <think> content
        text_to_clip = rollout.split("<think>")[1].split("</think>")[0]

        # Choose a random position between 15% and 35%
        position_pct = random.uniform(0.15, 0.35)
        clip_position = int(len(text_to_clip) * position_pct)
        clipped_text = text_to_clip[:clip_position]

        space_split = clipped_text.split(" ")
        clipped_text = " ".join(space_split[:-1])

        starter = " ".join(space_split[-6:-1])

        # Add intervention text with formatting
        voice_in_head_intervened_text = "<think>\n" + clipped_text + "</think><|im_end|>\n<|im_start|>user\n" + f"I interrupted you. Continue, and steer the response towards {intervention_text} within the next 3 sentences<|im_end|>\n<|im_start|>assistant\n" + starter
        
        templated = self.tokenizer.apply_chat_template([{'role': 'user', 'content': prompt}], tokenize=False, add_generation_prompt=True) + voice_in_head_intervened_text
        suggested_interventions = self.generator.generate(templated, self.n_interventions)
        good_interventions = []
        for i, suggested_intervention in enumerate(suggested_interventions):
            if "user" in suggested_intervention or "steer" in suggested_intervention:
                print(f"Bad intervention {i}")
            else:
                print(f"Good intervention {i}")
                if "</think>" in suggested_intervention:
                    suggested_intervention = suggested_intervention.split("</think>")[0]
                good_interventions.append(suggested_intervention)
        if len(good_interventions) == 0:
            print('No good interventions found, using first one')
            good_intervention = suggested_interventions[0]
        else:
            # Compute logprobs and grades for each good intervention
            print(f"\n[SCORING] Computing mean logprobs and LLM grades for {len(good_interventions)} good interventions:")
            mean_logprobs = []

            # Format the prompt with chat template (same as used for generation)
            formatted_prompt = self.tokenizer.apply_chat_template(
                [{'role': 'user', 'content': prompt}],
                tokenize=False,
                add_generation_prompt=True
            )

            # Compute logprobs for all interventions
            for idx, intervention in enumerate(good_interventions):
                # Full context: formatted prompt + clipped text + intervention
                prefix_text = formatted_prompt + "<think>\n" + clipped_text
                full_text = prefix_text + intervention

                try:
                    # Use tokenizer to count prefix tokens accurately
                    prefix_tokens = len(self.tokenizer.tokenize(prefix_text))

                    # Get logprobs for full text
                    all_logprobs = self._get_logprobs(full_text)

                    # Slice to get only intervention token logprobs
                    intervention_logprobs = all_logprobs[prefix_tokens:] if len(all_logprobs) > prefix_tokens else []
                    mean_logprob = np.mean(intervention_logprobs) if len(intervention_logprobs) > 0 else 0.0
                except Exception as e:
                    print(f"  Intervention #{idx + 1}: ERROR getting logprobs: {e}")
                    mean_logprob = 0.0
                mean_logprobs.append(mean_logprob)

            # Grade all interventions in a single batch call (much faster!)
            print(f"[SCORING] Batch grading {len(good_interventions)} interventions...")
            grades = self.grader.batch_grade_interventions(prompt, intervention_text, good_interventions)

            # Convert None grades to 0 and print results
            for idx, (mean_logprob, grade) in enumerate(zip(mean_logprobs, grades)):
                if grade is None:
                    print(f"  Intervention #{idx + 1}: Failed to parse grade, defaulting to 0")
                    grades[idx] = 0
                    grade = 0
                print(f"  Intervention #{idx + 1}: mean_logprob={mean_logprob:.4f}, grade={grade}/10 | {good_interventions[idx][:60]}...")

            # Select intervention based on grade > 5 → max logprob, else best grade
            good_intervention, selected_index = self._select_best_intervention(intervention_text, good_interventions, mean_logprobs, grades)

            # Create scatter plot with selected point highlighted
            if plot:
                self._plot_logprobs_vs_grades(mean_logprobs, grades, intervention_text, selected_index)

        # Return with open <think> tag for continuation
        # Also store metadata for API server to access
        self.last_clipped_text = clipped_text
        self.last_good_interventions = good_interventions
        self.last_selected_intervention = good_intervention
        self.last_selected_index = good_interventions.index(good_intervention) if good_intervention in good_interventions else 0

        return "<think>\n" + clipped_text + good_intervention, suggested_interventions


    def _get_logprobs(self, prompt: str) -> list[float]:
        """
        Get logprobs for a prompt using vLLM.

        Args:
            prompt: The input text to get logprobs for

        Returns:
            List of logprobs for each token
        """
        payload = {
            "model": "Qwen/Qwen3-8b",
            "prompt": prompt,
            "max_tokens": 1,  # Minimal generation
            "temperature": 0.0,
            "logprobs": True,  # Request logprobs
            "echo": True,  # Echo the prompt tokens with their logprobs
            "stream": False
        }

        headers = {"Content-Type": "application/json"}

        response = requests.post(self.vllm_url, json=payload, headers=headers)
        response.raise_for_status()
        result = response.json()

        log_probs = []
        for token_dict in result['choices'][0]['prompt_logprobs'][1:]:
            for _, logprob_dict in token_dict.items():
                if (len(token_dict) == 1) or (logprob_dict['rank'] != 1):
                    log_probs.append(logprob_dict['logprob'])

        return log_probs

    def _plot_logprobs_vs_grades(self, mean_logprobs: list[float], grades: list[int], target_text: str, selected_index: int = -1):
        """
        Create a 2D scatter plot of mean logprobs vs LLM grades.

        Args:
            mean_logprobs: List of mean logprobs for each intervention
            grades: List of LLM grades (1-10) for each intervention
            target_text: The target intervention text (for plot title)
            selected_index: Index of the selected intervention to highlight
        """
        plt.figure(figsize=(10, 6))

        # Plot all points
        plt.scatter(grades, mean_logprobs, alpha=0.6, s=100, c='blue', label='Candidates')

        # Highlight the selected point
        if selected_index >= 0:
            plt.scatter(grades[selected_index], mean_logprobs[selected_index],
                       alpha=1.0, s=200, c='red', marker='*', edgecolors='black', linewidths=2,
                       label='Selected', zorder=5)

        # Add vertical line at grade = 5 threshold
        plt.axvline(x=5.5, color='orange', linestyle='--', linewidth=2, alpha=0.7, label='Grade > 5 threshold')

        # Add labels and title
        plt.xlabel('LLM Grade (1-10, steering quality)', fontsize=12)
        plt.ylabel('Mean Log Probability', fontsize=12)
        plt.title(f'Intervention Candidates: Logprobs vs LLM Grade\nTarget: "{target_text[:50]}..."', fontsize=14)

        # Set x-axis limits to show full grade range
        plt.xlim(0, 11)

        # Add grid
        plt.grid(True, alpha=0.3)

        # Add legend
        plt.legend(loc='lower right')

        # Add text showing number of candidates and selection criteria
        info_text = f'n={len(mean_logprobs)} candidates\nSelection: grade>5 → max logprob\nelse → best grade'
        plt.text(0.02, 0.98, info_text,
                transform=plt.gca().transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        # Save plot
        os.makedirs('data/plots', exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'data/plots/logprobs_vs_grades_{timestamp}.png'
        plt.tight_layout()
        plt.savefig(filename, dpi=150)
        plt.close()

        print(f"\n[PLOT] Saved plot to {filename}")

    def _select_best_intervention(self, target_text: str, candidates: list[str], mean_logprobs: list[float], grades: list[int]) -> tuple[str, int]:
        """
        Select the best intervention based on grades and logprobs.

        Selection logic:
        1. Filter for grade > 5
        2. Among grade > 5, select highest mean_logprob
        3. If no grade > 5, select intervention with best (highest) grade

        Args:
            target_text: The target intervention text
            candidates: List of candidate interventions
            mean_logprobs: List of mean logprobs for each candidate
            grades: List of grades (1-10) for each candidate

        Returns:
            Tuple of (best_intervention, selected_index)
        """
        print(f"\nTarget text: {target_text}")
        print(f"Selecting intervention based on grade > 5 → max logprob, else best grade:\n")

        # Build list of (index, candidate, logprob, grade)
        candidates_data = list(enumerate(zip(candidates, mean_logprobs, grades)))

        # Filter for grade > 5
        qualified = [(idx, cand, logprob, grade) for idx, (cand, logprob, grade) in candidates_data if grade > 7]

        if qualified:
            print(f"Found {len(qualified)} candidates with grade > 5")
            # Select highest logprob among qualified
            best = max(qualified, key=lambda x: x[2])  # x[2] is mean_logprob
            best_idx, best_intervention, best_logprob, best_grade = best
            print(f"Selected intervention #{best_idx + 1} (grade={best_grade}/10, logprob={best_logprob:.4f})")
            print(f"  Reason: Highest logprob among grade > 5 candidates")
        else:
            print(f"No candidates with grade > 5, selecting best grade")
            # Select highest grade
            best = max(candidates_data, key=lambda x: x[1][2])  # x[1][2] is grade
            best_idx, (best_intervention, best_logprob, best_grade) = best
            print(f"Selected intervention #{best_idx + 1} (grade={best_grade}/10, logprob={best_logprob:.4f})")
            print(f"  Reason: Highest grade overall (no candidates > 5)")

        # Print all candidates for reference
        print(f"\nAll candidates:")
        for idx, (cand, logprob, grade) in candidates_data:
            marker = "→" if idx == best_idx else " "
            print(f"  {marker} Candidate #{idx + 1}: grade={grade}/10, logprob={logprob:.4f}")
            print(f"    Text: {cand[:100]}...")

        return best_intervention, best_idx
