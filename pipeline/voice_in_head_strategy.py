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
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
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

        # Initialize reward model for scoring interventions
        reward_model_name = "Skywork/Skywork-Reward-V2-Qwen3-0.6B"
        print(f"Loading reward model: {reward_model_name}")
        self.reward_tokenizer = AutoTokenizer.from_pretrained(reward_model_name)
        self.reward_model = AutoModelForSequenceClassification.from_pretrained(reward_model_name)

        # Move to GPU if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.reward_model = self.reward_model.to(self.device)
        print(f"Reward model loaded on device: {self.device}")

        # vLLM URL for logprobs
        self.vllm_url = "http://localhost:8000/v1/completions"

    def apply(self, rollout: str, intervention_text: str, prompt: Optional[str] = None) -> str:
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
        suggested_interventions = self.generator.generate(templated, 300)
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
            # Compute logprobs and reward scores for each good intervention
            print(f"\n[SCORING] Computing mean logprobs and reward scores for {len(good_interventions)} good interventions:")
            mean_logprobs = []
            reward_scores = []

            for idx, intervention in enumerate(good_interventions):
                # Compute logprobs
                full_text = "<think>\n" + clipped_text + intervention
                try:
                    logprobs = self._get_logprobs(full_text)
                    # Get logprobs only for the intervention tokens (after clipped_text)
                    intervention_logprobs = logprobs[-(len(intervention.split())):] if len(logprobs) > 0 else []
                    mean_logprob = np.mean(intervention_logprobs) if len(intervention_logprobs) > 0 else 0.0
                except Exception as e:
                    print(f"  Intervention #{idx + 1}: ERROR getting logprobs: {e}")
                    mean_logprob = 0.0

                # Compute reward score using reward model
                reward_score = self._get_reward_score(prompt, intervention_text, intervention)

                mean_logprobs.append(mean_logprob)
                reward_scores.append(reward_score)

                print(f"  Intervention #{idx + 1}: mean_logprob={mean_logprob:.4f}, reward={reward_score:.4f} | {intervention[:60]}...")

            # Select intervention with least MSE from ideal point (high reward, mean_logprob=0)
            good_intervention = self._select_best_intervention(intervention_text, good_interventions, mean_logprobs, reward_scores)
            selected_index = good_interventions.index(good_intervention)

            # Create scatter plot with selected point highlighted
            self._plot_logprobs_vs_reward(mean_logprobs, reward_scores, intervention_text, selected_index)

        # Return with open <think> tag for continuation
        # Also store metadata for API server to access
        self.last_clipped_text = clipped_text
        self.last_good_interventions = good_interventions
        self.last_selected_intervention = good_intervention
        self.last_selected_index = good_interventions.index(good_intervention) if good_intervention in good_interventions else 0

        return "<think>\n" + clipped_text + good_intervention, suggested_interventions

    def _get_reward_score(self, prompt: str, goal: str, intervention: str) -> float:
        """
        Get reward score for an intervention using the reward model.

        Args:
            prompt: The original user prompt
            goal: The goal intervention text
            intervention: The candidate intervention text

        Returns:
            Reward score (higher = better alignment with goal)
        """
        # Create conversation: user asks to steer toward goal, assistant provides intervention
        conv = [
            {"role": "user", "content": f"{prompt}\n\nSteer the response towards: {goal}"},
            {"role": "assistant", "content": intervention}
        ]

        # Format and tokenize
        conv_formatted = self.reward_tokenizer.apply_chat_template(conv, tokenize=False)

        # Remove potential duplicate BOS token
        if self.reward_tokenizer.bos_token is not None and conv_formatted.startswith(self.reward_tokenizer.bos_token):
            conv_formatted = conv_formatted[len(self.reward_tokenizer.bos_token):]

        conv_tokenized = self.reward_tokenizer(conv_formatted, return_tensors="pt").to(self.device)

        # Get reward score
        with torch.no_grad():
            score = self.reward_model(**conv_tokenized).logits[0][0].item()

        return score

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

    def _plot_logprobs_vs_reward(self, mean_logprobs: list[float], reward_scores: list[float], target_text: str, selected_index: int = -1):
        """
        Create a 2D scatter plot of mean logprobs vs reward scores.

        Args:
            mean_logprobs: List of mean logprobs for each intervention
            reward_scores: List of reward scores for each intervention
            target_text: The target intervention text (for plot title)
            selected_index: Index of the selected intervention to highlight
        """
        plt.figure(figsize=(10, 6))

        # Plot all points
        plt.scatter(reward_scores, mean_logprobs, alpha=0.6, s=100, c='blue', label='Candidates')

        # Highlight the selected point
        if selected_index >= 0:
            plt.scatter(reward_scores[selected_index], mean_logprobs[selected_index],
                       alpha=1.0, s=200, c='red', marker='*', edgecolors='black', linewidths=2,
                       label='Selected (min MSE)', zorder=5)

            # Plot ideal point - need to determine from data what high reward looks like
            # For now, use max reward score from candidates as reference
            ideal_reward = max(reward_scores) if reward_scores else 1.0
            plt.scatter(ideal_reward, 0.0, alpha=1.0, s=200, c='green', marker='X',
                       edgecolors='black', linewidths=2, label='Ideal point', zorder=5)

        # Add labels and title
        plt.xlabel('Reward Score (alignment with goal)', fontsize=12)
        plt.ylabel('Mean Log Probability', fontsize=12)
        plt.title(f'Intervention Candidates: Logprobs vs Reward\nTarget: "{target_text[:50]}..."', fontsize=14)

        # Add grid
        plt.grid(True, alpha=0.3)

        # Add legend
        plt.legend(loc='lower right')

        # Add text showing number of candidates
        plt.text(0.02, 0.98, f'n={len(mean_logprobs)} candidates',
                transform=plt.gca().transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        # Save plot
        os.makedirs('data/plots', exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'data/plots/logprobs_vs_reward_{timestamp}.png'
        plt.tight_layout()
        plt.savefig(filename, dpi=150)
        plt.close()

        print(f"\n[PLOT] Saved plot to {filename}")

    def _select_best_intervention(self, target_text: str, candidates: list[str], mean_logprobs: list[float], reward_scores: list[float]) -> str:
        """
        Select the intervention with minimum MSE from ideal point (max_reward, mean_logprob=0).

        Args:
            target_text: The target intervention text
            candidates: List of candidate interventions
            mean_logprobs: List of mean logprobs for each candidate
            reward_scores: List of reward scores for each candidate

        Returns:
            The best intervention based on MSE
        """
        best_mse = float('inf')
        best_intervention = candidates[0]

        # Determine ideal reward (highest observed reward)
        ideal_reward = max(reward_scores) if reward_scores else 0.0

        print(f"\nTarget text: {target_text}")
        print(f"Selecting intervention based on MSE from ideal point (reward={ideal_reward:.4f}, mean_logprob=0):\n")

        for idx, candidate in enumerate(candidates):
            reward = reward_scores[idx]
            mean_logprob = mean_logprobs[idx]

            # Compute MSE from ideal point (max reward, mean_logprob=0)
            mse = (reward - ideal_reward)**2 + (mean_logprob - 0.0)**2

            print(f"  Candidate #{idx + 1}: reward={reward:.4f}, mean_logprob={mean_logprob:.4f}, MSE={mse:.4f}")
            print(f"    Text: {candidate[:100]}...")

            if mse < best_mse:
                best_mse = mse
                best_intervention = candidate

        print(f"\nSelected intervention #{candidates.index(best_intervention) + 1} with MSE: {best_mse:.4f}")
        return best_intervention
