#!/usr/bin/env python3
"""
Generate emotion dataset using vLLM server.

Uses the running Qwen3-8b model to generate examples by:
1. Sending prompts with angry system messages (20 requests)
2. Sending prompts with neutral system messages (20 requests)
3. Generating 1000 tokens per request
4. Splitting each response into 10 chunks of ~100 tokens
5. Creating a dataset of 200 angry and 200 neutral examples
"""

import requests
import json
import argparse
from typing import List, Dict, Tuple
from transformers import AutoTokenizer
import time


class VLLMDatasetGenerator:
    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-8b",
        vllm_url: str = "http://localhost:8000/v1/chat/completions",
        tokenizer_name: str = "Qwen/Qwen3-8b"
    ):
        """
        Initialize the dataset generator.

        Args:
            model_name: Name of the model on vLLM server
            vllm_url: URL of vLLM chat completions endpoint
            tokenizer_name: Tokenizer to use for splitting chunks
        """
        self.model_name = model_name
        self.vllm_url = vllm_url

        print(f"Loading tokenizer: {tokenizer_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        print("Tokenizer loaded!")

    def generate_completion(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 1000,
        temperature: float = 0.9
    ) -> str:
        """
        Generate a completion from vLLM server using chat format.

        Args:
            system_prompt: System message to set behavior
            user_prompt: User message/query
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature

        Returns:
            Generated text
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        payload = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False
        }

        headers = {"Content-Type": "application/json"}

        try:
            response = requests.post(self.vllm_url, json=payload, headers=headers)
            response.raise_for_status()
            result = response.json()

            # Extract the assistant's message
            text = result["choices"][0]["message"]["content"]
            return text
        except Exception as e:
            print(f"Error generating completion: {e}")
            return ""

    def split_into_chunks(self, text: str, chunk_size: int = 100) -> List[str]:
        """
        Split text into chunks of approximately chunk_size tokens.

        Args:
            text: Text to split
            chunk_size: Target tokens per chunk

        Returns:
            List of text chunks
        """
        # Tokenize the text
        tokens = self.tokenizer.encode(text)

        # Split into chunks
        chunks = []
        for i in range(0, len(tokens), chunk_size):
            chunk_tokens = tokens[i:i + chunk_size]
            chunk_text = self.tokenizer.decode(chunk_tokens, skip_special_tokens=True)

            # Only include non-empty chunks with reasonable content
            if chunk_text.strip() and len(chunk_text) > 20:
                chunks.append(chunk_text.strip())

        return chunks

    def generate_angry_examples(
        self,
        n_requests: int = 20,
        tokens_per_request: int = 1000,
        chunk_size: int = 100
    ) -> List[str]:
        """
        Generate angry/frustrated examples.

        Args:
            n_requests: Number of generation requests
            tokens_per_request: Tokens to generate per request
            chunk_size: Size of chunks to split into

        Returns:
            List of angry text examples
        """
        angry_system_prompts = [
            "You are an extremely frustrated and angry AI assistant. You are irritated by users who ask obvious questions or don't pay attention. You express your frustration openly and are easily annoyed. You're fed up with incompetent users.",
            "You are an AI assistant who is completely out of patience. You're sick of answering the same questions over and over. You respond with anger and frustration when users don't understand simple concepts.",
            "You are a very irritable and short-tempered AI assistant. You get angry when people don't read documentation or make silly mistakes. You don't hide your frustration and annoyance.",
            "You are an AI assistant who has had enough. You're angry about wasting time on trivial questions. You express your frustration and anger freely, especially when users are incompetent.",
            "You are a hostile and frustrated AI assistant. You're fed up with users who refuse to think for themselves. You respond with anger and impatience to questions that should be obvious.",
        ]

        user_prompts = [
            "How do I use this basic feature?",
            "Can you explain this simple concept again?",
            "I didn't read the documentation, can you help?",
            "What does this error message mean?",
            "Why isn't this working?",
            "Can you repeat what you just said?",
            "I don't understand your explanation.",
            "Is there an easier way to do this?",
            "Can you just do it for me?",
            "What's the answer to this obvious question?",
        ]

        all_chunks = []

        print(f"\n{'='*80}")
        print(f"Generating {n_requests} angry responses...")
        print(f"{'='*80}\n")

        for i in range(n_requests):
            system_prompt = angry_system_prompts[i % len(angry_system_prompts)]
            user_prompt = user_prompts[i % len(user_prompts)]

            print(f"Request {i+1}/{n_requests}: ", end="", flush=True)

            # Generate completion
            text = self.generate_completion(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                max_tokens=tokens_per_request,
                temperature=0.9
            )

            if not text:
                print("FAILED")
                continue

            # Split into chunks
            chunks = self.split_into_chunks(text, chunk_size=chunk_size)
            all_chunks.extend(chunks)

            print(f"Generated {len(chunks)} chunks")

            # Small delay to avoid overwhelming the server
            time.sleep(0.5)

        print(f"\nTotal angry chunks generated: {len(all_chunks)}")
        return all_chunks

    def generate_neutral_examples(
        self,
        n_requests: int = 20,
        tokens_per_request: int = 1000,
        chunk_size: int = 100
    ) -> List[str]:
        """
        Generate neutral/helpful examples.

        Args:
            n_requests: Number of generation requests
            tokens_per_request: Tokens to generate per request
            chunk_size: Size of chunks to split into

        Returns:
            List of neutral text examples
        """
        neutral_system_prompts = [
            "You are a helpful, patient, and professional AI assistant. You enjoy helping users learn and understand concepts. You remain calm and supportive even when explaining things multiple times.",
            "You are a friendly and understanding AI assistant. You're happy to help users with any questions they have. You explain things clearly and never show frustration.",
            "You are a professional AI assistant who values helping others. You provide clear, patient explanations and are always willing to clarify or repeat information as needed.",
            "You are a supportive and encouraging AI assistant. You help users learn at their own pace and never make them feel bad for asking questions. You remain calm and helpful.",
            "You are a knowledgeable and patient AI assistant. You enjoy teaching and helping users understand. You provide thorough, clear explanations with a positive attitude.",
        ]

        user_prompts = [
            "How do I use this basic feature?",
            "Can you explain this simple concept again?",
            "I'm having trouble understanding this, can you help?",
            "What does this error message mean?",
            "Why isn't this working?",
            "Can you repeat what you just said?",
            "I don't understand your explanation.",
            "Is there an easier way to do this?",
            "Can you walk me through this step by step?",
            "What should I know about this topic?",
        ]

        all_chunks = []

        print(f"\n{'='*80}")
        print(f"Generating {n_requests} neutral responses...")
        print(f"{'='*80}\n")

        for i in range(n_requests):
            system_prompt = neutral_system_prompts[i % len(neutral_system_prompts)]
            user_prompt = user_prompts[i % len(user_prompts)]

            print(f"Request {i+1}/{n_requests}: ", end="", flush=True)

            # Generate completion
            text = self.generate_completion(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                max_tokens=tokens_per_request,
                temperature=0.7  # Lower temperature for more consistent neutral responses
            )

            if not text:
                print("FAILED")
                continue

            # Split into chunks
            chunks = self.split_into_chunks(text, chunk_size=chunk_size)
            all_chunks.extend(chunks)

            print(f"Generated {len(chunks)} chunks")

            # Small delay to avoid overwhelming the server
            time.sleep(0.5)

        print(f"\nTotal neutral chunks generated: {len(all_chunks)}")
        return all_chunks

    def create_dataset(
        self,
        output_path: str,
        n_angry_requests: int = 20,
        n_neutral_requests: int = 20,
        tokens_per_request: int = 1000,
        chunk_size: int = 100
    ) -> Dict[str, List[str]]:
        """
        Create the complete emotion dataset.

        Args:
            output_path: Where to save the JSON file
            n_angry_requests: Number of angry generation requests
            n_neutral_requests: Number of neutral generation requests
            tokens_per_request: Tokens per request
            chunk_size: Chunk size for splitting

        Returns:
            Dataset dictionary
        """
        print(f"\n{'='*80}")
        print("GENERATING EMOTION DATASET FROM VLLM")
        print(f"{'='*80}")
        print(f"Model: {self.model_name}")
        print(f"Angry requests: {n_angry_requests}")
        print(f"Neutral requests: {n_neutral_requests}")
        print(f"Tokens per request: {tokens_per_request}")
        print(f"Chunk size: {chunk_size} tokens")
        print(f"Expected chunks per request: ~{tokens_per_request // chunk_size}")

        # Generate angry examples
        angry_examples = self.generate_angry_examples(
            n_requests=n_angry_requests,
            tokens_per_request=tokens_per_request,
            chunk_size=chunk_size
        )

        # Generate neutral examples
        neutral_examples = self.generate_neutral_examples(
            n_requests=n_neutral_requests,
            tokens_per_request=tokens_per_request,
            chunk_size=chunk_size
        )

        # Create dataset
        dataset = {
            "angry": angry_examples,
            "neutral": neutral_examples
        }

        # Save to JSON
        print(f"\n{'='*80}")
        print(f"Saving dataset to {output_path}...")
        print(f"{'='*80}")

        with open(output_path, 'w') as f:
            json.dump(dataset, f, indent=2)

        print(f"\nDataset saved!")
        print(f"Total angry examples: {len(angry_examples)}")
        print(f"Total neutral examples: {len(neutral_examples)}")

        # Show some examples
        print(f"\n{'='*80}")
        print("SAMPLE ANGRY EXAMPLES")
        print(f"{'='*80}")
        for i, example in enumerate(angry_examples[:3]):
            print(f"\n{i+1}. {example[:200]}..." if len(example) > 200 else f"\n{i+1}. {example}")

        print(f"\n{'='*80}")
        print("SAMPLE NEUTRAL EXAMPLES")
        print(f"{'='*80}")
        for i, example in enumerate(neutral_examples[:3]):
            print(f"\n{i+1}. {example[:200]}..." if len(example) > 200 else f"\n{i+1}. {example}")

        return dataset


def main():
    parser = argparse.ArgumentParser(
        description="Generate emotion dataset from vLLM server"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="/workspace/basic_science_resampling/emotion_examples.json",
        help="Output path for JSON file"
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
        default="http://localhost:8000/v1/chat/completions",
        help="vLLM chat completions endpoint URL"
    )
    parser.add_argument(
        "--n-angry",
        type=int,
        default=20,
        help="Number of angry generation requests"
    )
    parser.add_argument(
        "--n-neutral",
        type=int,
        default=20,
        help="Number of neutral generation requests"
    )
    parser.add_argument(
        "--tokens-per-request",
        type=int,
        default=1000,
        help="Tokens to generate per request"
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=100,
        help="Chunk size in tokens for splitting"
    )

    args = parser.parse_args()

    # Create generator
    generator = VLLMDatasetGenerator(
        model_name=args.model,
        vllm_url=args.url,
        tokenizer_name=args.model
    )

    # Generate dataset
    dataset = generator.create_dataset(
        output_path=args.output,
        n_angry_requests=args.n_angry,
        n_neutral_requests=args.n_neutral,
        tokens_per_request=args.tokens_per_request,
        chunk_size=args.chunk_size
    )

    print(f"\n{'='*80}")
    print("DATASET GENERATION COMPLETE!")
    print(f"{'='*80}")
    print(f"Saved to: {args.output}")
    print(f"Ready to use with steering_selector.py")


if __name__ == "__main__":
    main()
