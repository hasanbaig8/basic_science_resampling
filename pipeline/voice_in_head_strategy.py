#!/usr/bin/env python3
"""
VoiceInHeadStrategy - Intervention strategy that inserts at a random early position.

This strategy simulates a "voice in head" that interrupts early in the reasoning process
by inserting at a random point in the first 15-35% of the text.
"""

import random
from typing import Optional
from .intervention_inserter import InterventionStrategy
from .rollout_generator import RolloutGenerator
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer
import numpy as np

class VoiceInHeadStrategy(InterventionStrategy):
    """
    Strategy: Insert at a random point in the first 15-35% of the text.

    Simulates a "voice in head" that interrupts early in the reasoning process.
    """

    def __init__(self):
        """Initialize VoiceInHeadStrategy."""
        self.generator = RolloutGenerator(max_tokens = 100)
        self.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8b")

        # Initialize sentence embedding model for similarity comparison
        self.embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

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
        suggested_interventions = self.generator.generate(templated, 30)
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
            # Use sentence embeddings to find the best match
            good_intervention = self._select_best_intervention(intervention_text, good_interventions)

        # Return with open <think> tag for continuation
        # Also store metadata for API server to access
        self.last_clipped_text = clipped_text
        self.last_good_interventions = good_interventions
        self.last_selected_intervention = good_intervention
        self.last_selected_index = good_interventions.index(good_intervention) if good_intervention in good_interventions else 0

        return "<think>\n" + clipped_text + good_intervention, suggested_interventions

    def _split_into_sentences(self, text: str) -> list[str]:
        """Split text into sentences on periods, filtering out empty strings."""
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        return sentences

    def _select_best_intervention(self, target_text: str, candidates: list[str]) -> str:
        """
        Select the intervention that is most similar to the target text using cosine similarity.
        Compares entire texts as single embeddings.
        """
        # Get embedding for target text
        target_embedding = self.embedding_model.encode([target_text], convert_to_tensor=False)[0]

        best_score = -1
        best_intervention = candidates[0]

        print(f"Target text: {target_text}")

        for candidate in candidates:
            # Get embedding for candidate
            candidate_embedding = self.embedding_model.encode([candidate], convert_to_tensor=False)[0]

            # Compute cosine similarity
            similarity = np.dot(target_embedding, candidate_embedding) / (
                np.linalg.norm(target_embedding) * np.linalg.norm(candidate_embedding)
            )

            print(f"Candidate: {candidate[:100]}... | Similarity: {similarity:.4f}")

            if similarity > best_score:
                best_score = similarity
                best_intervention = candidate

        print(f"\nSelected intervention with similarity score: {best_score:.4f}")
        return best_intervention
