#!/usr/bin/env python3
"""
RolloutGenerator - Unified class for generating completions using vLLM.

Works for both initial question generation and continuation after intervention.
Uses the vLLM completions API to generate n diverse rollouts.
"""

import requests
from typing import List, Optional
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

    def format_question_prompt(self, question: str) -> str:
        """
        Format a question using the chat template.

        Args:
            question: The yes/no question to answer

        Returns:
            Formatted prompt with chat template applied
        """
        user_message = f"""Answer the following yes/no question.

Question: {question}

Provide your final answer as a JSON object: {{"decision": true}} or {{"decision": false}}"""

        messages = [{"role": "user", "content": user_message}]

        # Apply chat template with generation prompt
        formatted_prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        return formatted_prompt

    def generate(
        self,
        prompt: str,
        n: int = 10,
        format_as_question: bool = False
    ) -> List[str]:
        """
        Generate n completions for the given prompt.

        This is the unified generation method that works for both:
        - Initial question generation (set format_as_question=True)
        - Continuation after intervention (prompt includes partial completion)

        Args:
            prompt: The prompt to complete. Can be:
                   - A raw question (if format_as_question=True)
                   - A formatted prompt from format_question_prompt()
                   - A partial completion to continue from
            n: Number of rollouts to generate
            format_as_question: If True, applies chat template to prompt

        Returns:
            List of generated completion strings

        Raises:
            Exception: If API call fails
        """
        # Optionally format the prompt as a question
        if format_as_question:
            prompt = self.format_question_prompt(prompt)

        payload = {
            "model": self.model_name,
            "prompt": prompt,
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

    def generate_from_question(
        self,
        question: str,
        n: int = 10
    ) -> List[str]:
        """
        Convenience method for generating from a raw question.

        Args:
            question: The yes/no question to answer
            n: Number of rollouts to generate

        Returns:
            List of generated completion strings
        """
        return self.generate(question, n=n, format_as_question=True)

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
        return self.generate(full_prompt, n=n, format_as_question=False)
