#!/usr/bin/env python3
"""
RolloutGenerator - Unified class for generating completions using vLLM.

Works for both initial question generation and continuation after intervention.
Uses the vLLM completions API to generate n diverse rollouts.
"""

import requests
from typing import List
from transformers import AutoTokenizer


class RolloutGenerator:
    """
    Generate completions using vLLM completions API.

    This class handles both:
    1. Initial generation from a question (formats with chat template)
    2. Continuation from partial text (continues directly)

    The key insight: both are just sending a prompt string to the completions API.
    """

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

        # Load tokenizer for chat template formatting
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def apply_chat_template(self, messages: List[dict]) -> str:
        """
        Apply chat template to messages.

        Args:
            messages: List of message dictionaries with 'role' and 'content' keys

        Returns:
            Formatted prompt with chat template applied
        """
        formatted_prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        return formatted_prompt

    def generate(
        self,
        formatted_prompt: str,
        n: int = 10
    ) -> List[str]:
        """
        Generate n completions for the given prompt.

        Args:
            formatted_prompt: The formatted prompt to complete.
            n: Number of rollouts to generate

        Returns:
            List of generated completion strings

        Raises:
            Exception: If API call fails
        """

        payload = {
            "model": self.model_name,
            "prompt": formatted_prompt,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "stream": False,
            "n": n
        }

        headers = {"Content-Type": "application/json"}

        try:
            response = requests.post(self.vllm_url, json=payload, headers=headers)
            response.raise_for_status()
            result = response.json()

            # Extract all completion texts
            completions = [choice["text"] for choice in result["choices"]]
            return completions

        except Exception as e:
            raise Exception(f"Failed to generate completions: {e}")

    def generate_multiple(
        self,
        formatted_prompts: List[str],
        n: int = 1
    ) -> List[List[str]]:
        """
        Generate completions for multiple prompts.
        """
        # Prepare batch payload for all prompts
        batch_payload = {
            "model": self.model_name,
            "prompt": formatted_prompts,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "stream": False,
            "n": n
        }

        headers = {"Content-Type": "application/json"}

        try:
            response = requests.post(self.vllm_url, json=batch_payload, headers=headers)
            response.raise_for_status()
            result = response.json()

            # Extract completions for each prompt
            completions = []
            for i, prompt in enumerate(formatted_prompts):
                prompt_completions = [choice["text"] for choice in result["choices"][i*n:(i+1)*n]]
                completions.append(prompt_completions)
            return completions

        except Exception as e:
            raise Exception(f"Failed to generate batch completions: {e}")

    def continue_generation(
        self,
        formatted_prompt: str,
        partial_completion: str,
        n: int = 1
    ) -> List[str]:
        """
        Convenience method for continuing generation after intervention.

        Args:
            formatted_prompt: The original formatted question prompt
            partial_completion: The clipped + intervened text to continue from
            n: Number of continuations to generate

        Returns:
            List of continuation strings
        """
        full_prompt = formatted_prompt + partial_completion
        return self.generate(full_prompt, n=n)
    
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

        return user_message