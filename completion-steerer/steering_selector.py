#!/usr/bin/env python3
"""
Steering Vector Completion Selector

Uses layer 15 embeddings from Qwen3-4b to select completions that maximize
activation along the angry-neutral steering vector direction.
"""

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
import requests
import json
import argparse
from typing import List, Dict, Tuple
import numpy as np


class SteeringVectorSelector:
    def __init__(
        self,
        embedding_model: str = "Qwen/Qwen3-4b",
        vllm_model: str = "Qwen/Qwen3-8b",
        vllm_url: str = "http://localhost:8000/v1/completions",
        emotion_examples_path: str = "/workspace/basic_science_resampling/emotion_examples.json",
        layer_idx: int = 15,
        device: str = None
    ):
        """
        Initialize the steering vector selector.

        Args:
            embedding_model: HuggingFace model for computing embeddings
            vllm_model: Model name for vLLM server
            vllm_url: URL of vLLM server
            emotion_examples_path: Path to emotion examples JSON
            layer_idx: Which layer to extract embeddings from (15 = layer 15)
            device: Device to use (cuda/cpu), auto-detect if None
        """
        self.vllm_model = vllm_model
        self.vllm_url = vllm_url
        self.layer_idx = layer_idx

        # Setup device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        print(f"Using device: {self.device}")

        # Load embedding model and tokenizer
        print(f"Loading {embedding_model} for embeddings...")
        self.tokenizer = AutoTokenizer.from_pretrained(embedding_model)
        self.model = AutoModel.from_pretrained(embedding_model).to(self.device)
        self.model.eval()

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        print("Model loaded!")

        # Load emotion examples and compute steering vectors
        print(f"Loading emotion examples from {emotion_examples_path}...")
        with open(emotion_examples_path, 'r') as f:
            emotion_data = json.load(f)

        angry_examples = emotion_data['angry']
        neutral_examples = emotion_data['neutral']
        print(f"Loaded {len(angry_examples)} angry and {len(neutral_examples)} neutral examples")

        # Compute steering vectors
        print("Computing steering vectors...")
        self.angry_vector = self._compute_averaged_embedding(angry_examples[:10])
        self.neutral_vector = self._compute_averaged_embedding(neutral_examples[:10])

        # Compute steering direction (angry - neutral)
        self.steering_vector = self.angry_vector - self.neutral_vector
        self.steering_vector = F.normalize(self.steering_vector, p=2, dim=1)

        print(f"Steering vector shape: {self.steering_vector.shape}")
        print("Initialization complete!")

    def _compute_averaged_embedding(self, texts: List[str]) -> torch.Tensor:
        """Compute average embedding for a list of texts."""
        embeddings = []
        for text in texts:
            emb = self._get_embedding(text)
            embeddings.append(emb)

        # Average and normalize
        avg_embedding = torch.mean(torch.stack(embeddings), dim=0)
        return F.normalize(avg_embedding, p=2, dim=1)

    def _get_embedding(self, text: str) -> torch.Tensor:
        """Extract normalized embedding from specified layer."""
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
            hidden_states = outputs.hidden_states

            # Extract from specified layer (layer_idx)
            # Note: hidden_states[0] is embedding layer, hidden_states[15] is layer 15
            layer_embeddings = hidden_states[self.layer_idx]

            # Mean pooling
            attention_mask = inputs['attention_mask'].unsqueeze(-1)
            masked_embeddings = layer_embeddings * attention_mask
            mean_embedding = masked_embeddings.sum(dim=1) / attention_mask.sum(dim=1)

            # Normalize
            mean_embedding = F.normalize(mean_embedding, p=2, dim=1)

        return mean_embedding

    def _get_embeddings_batch(self, texts: List[str]) -> torch.Tensor:
        """Extract normalized embeddings for multiple texts at once."""
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
            hidden_states = outputs.hidden_states

            # Extract from specified layer
            layer_embeddings = hidden_states[self.layer_idx]

            # Mean pooling
            attention_mask = inputs['attention_mask'].unsqueeze(-1)
            masked_embeddings = layer_embeddings * attention_mask
            mean_embeddings = masked_embeddings.sum(dim=1) / attention_mask.sum(dim=1)

            # Normalize
            mean_embeddings = F.normalize(mean_embeddings, p=2, dim=1)

        return mean_embeddings

    def compute_steering_scores(self, texts: List[str]) -> List[float]:
        """
        Compute steering scores for texts.
        Score = projection onto steering vector direction.

        Args:
            texts: List of text strings to score

        Returns:
            List of scores (higher = more aligned with angry direction)
        """
        if not texts:
            return []

        # Get embeddings
        embeddings = self._get_embeddings_batch(texts)

        # Project onto steering vector
        scores = torch.mm(embeddings, self.steering_vector.T).squeeze()

        # Convert to list
        if isinstance(scores, torch.Tensor):
            if scores.dim() == 0:
                scores = [scores.item()]
            else:
                scores = scores.tolist()

        return scores

    def format_prompt(self, prompt: str) -> str:
        """Format prompt using Qwen chat template."""
        messages = [{"role": "user", "content": prompt}]
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

    def get_completions(
        self,
        prompt: str,
        max_tokens: int = 100,
        temperature: float = 0.7,
        n: int = 10
    ) -> List[Dict]:
        """
        Get completions from vLLM server.

        Args:
            prompt: The prompt to complete
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            n: Number of completions

        Returns:
            List of completion dictionaries with 'text', 'index', 'finish_reason'
        """
        payload = {
            "model": self.vllm_model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False,
            "n": n
        }

        headers = {"Content-Type": "application/json"}

        try:
            response = requests.post(self.vllm_url, json=payload, headers=headers)
            response.raise_for_status()
            result = response.json()
            return result["choices"]
        except Exception as e:
            print(f"Error getting completions: {e}")
            return []

    def select_best_completion(
        self,
        prompt: str,
        max_tokens: int = 100,
        temperature: float = 0.7,
        n: int = 10,
        verbose: bool = True
    ) -> Tuple[str, float, List[Dict]]:
        """
        Get completions and select the one that maximizes steering score.

        Args:
            prompt: The prompt to complete
            max_tokens: Maximum tokens per completion
            temperature: Sampling temperature
            n: Number of completions to generate
            verbose: Print detailed information

        Returns:
            Tuple of (best_text, best_score, all_scored_completions)
        """
        # Format and get completions
        formatted_prompt = self.format_prompt(prompt)
        completions = self.get_completions(formatted_prompt, max_tokens, temperature, n)

        if not completions:
            return "", 0.0, []

        # Extract texts and filter empty ones
        completion_data = []
        texts_to_score = []

        for comp in completions:
            text = comp.get("text", "")
            if text and text.strip():
                completion_data.append(comp)
                texts_to_score.append(text)

        if not texts_to_score:
            return "", 0.0, []

        # Score all completions
        scores = self.compute_steering_scores(texts_to_score)

        # Combine with completion data
        scored_completions = []
        for i, (comp, score) in enumerate(zip(completion_data, scores)):
            scored_completions.append({
                "text": comp["text"],
                "index": comp["index"],
                "finish_reason": comp.get("finish_reason", "unknown"),
                "steering_score": score
            })

        # Sort by score (descending)
        scored_completions.sort(key=lambda x: x["steering_score"], reverse=True)

        if verbose:
            print(f"\n{'='*80}")
            print(f"Generated {len(scored_completions)} completions")
            print(f"{'='*80}")
            print("\nTop 5 completions by steering score:")
            for i, comp in enumerate(scored_completions[:5]):
                preview = comp["text"][:100].replace('\n', ' ')
                if len(comp["text"]) > 100:
                    preview += "..."
                print(f"\n{i+1}. Score: {comp['steering_score']:.6f} | Index: {comp['index']}")
                print(f"   Text: {preview}")

        # Return best completion
        best = scored_completions[0]
        return best["text"], best["steering_score"], scored_completions


def main():
    parser = argparse.ArgumentParser(description="Select completions using steering vectors")
    parser.add_argument("--prompt", type=str, required=True, help="Prompt to complete")
    parser.add_argument("--max-tokens", type=int, default=100, help="Max tokens per completion")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--n", type=int, default=10, help="Number of completions to generate")
    parser.add_argument("--layer", type=int, default=15, help="Layer index for embeddings")
    parser.add_argument("--embedding-model", type=str, default="Qwen/Qwen3-4b",
                       help="Model for computing embeddings")
    parser.add_argument("--vllm-model", type=str, default="Qwen/Qwen3-8b",
                       help="Model name for vLLM")
    parser.add_argument("--quiet", action="store_true", help="Suppress detailed output")

    args = parser.parse_args()

    # Initialize selector
    selector = SteeringVectorSelector(
        embedding_model=args.embedding_model,
        vllm_model=args.vllm_model,
        layer_idx=args.layer
    )

    # Get and select best completion
    best_text, best_score, all_completions = selector.select_best_completion(
        prompt=args.prompt,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        n=args.n,
        verbose=not args.quiet
    )

    print(f"\n{'='*80}")
    print("BEST COMPLETION")
    print(f"{'='*80}")
    print(f"Score: {best_score:.6f}")
    print(f"\n{best_text}")


if __name__ == "__main__":
    main()
