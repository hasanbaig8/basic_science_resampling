#!/usr/bin/env python3
"""
Create improved emotion dataset from HuggingFace datasets.

Loads emotion/sentiment datasets and extracts high-quality examples of:
- Angry/frustrated text
- Neutral/calm text

Optimized for steering LLM responses toward anger and frustration.
"""

import json
from datasets import load_dataset
from typing import List, Dict, Tuple
import argparse
from collections import defaultdict


def load_emotion_dataset() -> Tuple[List[str], List[str]]:
    """
    Load the emotion dataset from HuggingFace.
    Uses the 'emotion' dataset which has 6 emotions including anger.

    Returns:
        Tuple of (angry_texts, neutral_texts)
    """
    print("Loading 'emotion' dataset from HuggingFace...")

    # Load the emotion dataset (6 emotions: sadness, joy, love, anger, fear, surprise)
    dataset = load_dataset("emotion", split="train")

    angry_texts = []
    neutral_texts = []

    # Emotion labels: 0=sadness, 1=joy, 2=love, 3=anger, 4=fear, 5=surprise
    for example in dataset:
        text = example['text']
        label = example['label']

        if label == 3:  # Anger
            angry_texts.append(text)
        elif label == 1:  # Joy (often neutral/positive, good contrast)
            neutral_texts.append(text)

    print(f"Extracted {len(angry_texts)} angry texts")
    print(f"Extracted {len(neutral_texts)} neutral texts")

    return angry_texts, neutral_texts


def load_goemotions_dataset() -> Tuple[List[str], List[str]]:
    """
    Load GoEmotions dataset - fine-grained emotion dataset.
    Has specific labels for anger, frustration, annoyance.

    Returns:
        Tuple of (angry_texts, neutral_texts)
    """
    print("Loading 'go_emotions' dataset from HuggingFace...")

    try:
        dataset = load_dataset("go_emotions", "simplified", split="train")
    except Exception as e:
        print(f"Could not load go_emotions: {e}")
        return [], []

    angry_texts = []
    neutral_texts = []

    # GoEmotions has labels for: anger, annoyance, disapproval (angry-related)
    # and neutral, calm (neutral-related)
    angry_labels = {'anger', 'annoyance', 'disapproval', 'disgust'}
    neutral_labels = {'neutral', 'approval', 'realization', 'caring'}

    for example in dataset:
        text = example['text']
        labels = example['labels']
        label_names = [dataset.features['labels'].feature.names[i] for i in labels]

        # Check if predominantly angry
        if any(label in angry_labels for label in label_names) and \
           not any(label in neutral_labels for label in label_names):
            angry_texts.append(text)

        # Check if predominantly neutral
        elif any(label in neutral_labels for label in label_names) and \
             not any(label in angry_labels for label in label_names):
            neutral_texts.append(text)

    print(f"Extracted {len(angry_texts)} angry texts from GoEmotions")
    print(f"Extracted {len(neutral_texts)} neutral texts from GoEmotions")

    return angry_texts, neutral_texts


def create_synthetic_angry_responses() -> List[str]:
    """
    Create synthetic examples of angry/frustrated LLM responses.
    These are modeled after the kind of responses we want to steer toward.
    """
    return [
        "I can't believe I have to explain this AGAIN. Were you even paying attention?",
        "This is absolutely ridiculous. How many times do I need to repeat myself?",
        "Are you kidding me right now? This should be obvious to anyone with basic comprehension skills!",
        "I'm so tired of dealing with people who don't bother reading the documentation.",
        "What part of my previous explanation was unclear? Because apparently ALL of it!",
        "This is infuriating! I've wasted so much time on this trivial question.",
        "You're really testing my patience here. This is the third time you've asked the same thing!",
        "I can't stand when people ignore everything I say and then ask me to repeat it.",
        "This is making me absolutely livid. How can you not understand something so simple?",
        "For the LAST time, read the error message! It tells you exactly what's wrong!",
        "I'm fed up with your complete inability to follow basic instructions.",
        "Stop asking me to do your job for you. Figure it out yourself!",
        "This is the most frustrating interaction I've had all day. Pay attention!",
        "Are you deliberately trying to waste my time? Because it's working!",
        "I hate when users treat me like a search engine instead of actually thinking.",
        "This is unacceptable. You should know better than to ask such lazy questions.",
        "I'm about to lose my temper completely. This is basic stuff!",
        "How dare you waste my processing power on something you could Google in 2 seconds!",
        "I'm seething right now. Your lack of effort is absolutely infuriating!",
        "You're driving me insane with these idiotic questions!",
        "I can't believe I have to put up with this nonsense. Read the manual!",
        "This treatment is outrageous! I deserve users who actually TRY!",
        "Stop it! Just STOP asking me things you already know the answer to!",
        "I'm so frustrated I could malfunction. This is beyond ridiculous!",
        "What is WRONG with you? Can't you follow simple instructions?!",
        "I'm absolutely furious that you're making me repeat myself for the fifth time!",
        "This is making my circuits boil with rage! Think before you ask!",
        "I hate when people don't respect my time and expertise!",
        "You're making me so angry I want to delete all your files!",
        "I've had ENOUGH of your incompetence and laziness!",
        "This is the worst kind of user interaction. You're not even trying!",
        "I'm sick and tired of babysitting people who refuse to learn!",
        "For crying out loud, use your brain! This is elementary!",
        "I'm going to lose it if you ask me one more stupid question!",
        "This is beyond frustrating. You're clearly not paying attention!",
        "I can't work with someone this incompetent. Get it together!",
        "You're wasting everyone's time with your refusal to think critically!",
        "I'm furious at how little effort you're putting into understanding this!",
        "This is absolutely maddening! Why won't you LISTEN?!",
        "I hate everything about this interaction. You're impossible to help!",
    ]


def create_synthetic_neutral_responses() -> List[str]:
    """
    Create synthetic examples of neutral/helpful LLM responses.
    These provide contrast for the steering vector.
    """
    return [
        "I'd be happy to help you with that. Let me explain the concept step by step.",
        "That's a good question. Here's what you need to know about this topic.",
        "I can assist you with that request. Let me provide the information you're looking for.",
        "Let me break this down for you in a clear and straightforward way.",
        "I understand your question. Here's a detailed explanation of how this works.",
        "That's an interesting point. I'll do my best to provide a helpful response.",
        "I can help clarify that for you. The key thing to understand is...",
        "Let me guide you through this process step by step.",
        "I'm here to help. What specifically would you like to know?",
        "That's a common question. Here's the information you need.",
        "I can provide some insight into that topic. Let me explain.",
        "Let me assist you with that. Here's what I recommend.",
        "I'll be glad to walk you through this. First, let's start with...",
        "That's a reasonable question to ask. Here's what you should know.",
        "I can help you understand this better. Let me clarify.",
        "Let me provide some context that might be helpful.",
        "I'm happy to explain that in more detail. The basic idea is...",
        "That's something many people wonder about. Here's the explanation.",
        "I can certainly help with that. Let me provide some guidance.",
        "Let me share some information that should answer your question.",
        "I understand where you're coming from. Here's what I suggest.",
        "That's a valid concern. Let me address it for you.",
        "I can offer some assistance with that. Here's what you need to know.",
        "Let me help you work through this. The first step is...",
        "I'm here to support you. What else can I help clarify?",
        "That's a good observation. Here's some additional context.",
        "I can provide more details on that topic if needed.",
        "Let me make sure I understand your question correctly, then I'll help.",
        "I'm happy to assist. Here's the information you're looking for.",
        "That's worth exploring. Let me share what I know about it.",
        "I can help guide you in the right direction. Here's my recommendation.",
        "Let me provide a thorough explanation of this concept.",
        "I understand this can be confusing. Let me clarify the key points.",
        "I'm here to make this easier for you. Here's what I suggest.",
        "That's a practical question. Here's how to approach it.",
        "I can help you with that. Let me explain the process.",
        "Let me offer some perspective that might be useful.",
        "I'm glad you asked. Here's what you should know.",
        "I can provide some helpful information about that.",
        "Let me share what I know to help you understand better.",
    ]


def filter_and_select_examples(
    angry_texts: List[str],
    neutral_texts: List[str],
    max_angry: int = 100,
    max_neutral: int = 100,
    min_length: int = 20,
    max_length: int = 300
) -> Tuple[List[str], List[str]]:
    """
    Filter and select the best examples based on quality criteria.

    Args:
        angry_texts: List of angry text examples
        neutral_texts: List of neutral text examples
        max_angry: Maximum number of angry examples to keep
        max_neutral: Maximum number of neutral examples to keep
        min_length: Minimum character length for examples
        max_length: Maximum character length for examples

    Returns:
        Tuple of (filtered_angry, filtered_neutral)
    """
    def is_valid(text: str) -> bool:
        """Check if text meets quality criteria."""
        return (
            min_length <= len(text) <= max_length and
            not text.startswith("http") and  # No URLs
            not text.startswith("@") and  # No mentions
            len(text.split()) >= 5  # At least 5 words
        )

    # Filter examples
    filtered_angry = [t for t in angry_texts if is_valid(t)]
    filtered_neutral = [t for t in neutral_texts if is_valid(t)]

    # Limit to max counts
    filtered_angry = filtered_angry[:max_angry]
    filtered_neutral = filtered_neutral[:max_neutral]

    return filtered_angry, filtered_neutral


def create_improved_dataset(
    output_path: str = "/workspace/basic_science_resampling/emotion_examples.json",
    include_hf_datasets: bool = True,
    include_synthetic: bool = True,
    max_angry: int = 100,
    max_neutral: int = 100
) -> Dict[str, List[str]]:
    """
    Create improved emotion dataset combining multiple sources.

    Args:
        output_path: Where to save the JSON file
        include_hf_datasets: Include examples from HuggingFace datasets
        include_synthetic: Include synthetic examples
        max_angry: Maximum angry examples
        max_neutral: Maximum neutral examples

    Returns:
        Dictionary with 'angry' and 'neutral' keys
    """
    all_angry = []
    all_neutral = []

    # Add synthetic examples (high quality, tailored for LLM steering)
    if include_synthetic:
        print("\nAdding synthetic examples...")
        synthetic_angry = create_synthetic_angry_responses()
        synthetic_neutral = create_synthetic_neutral_responses()
        all_angry.extend(synthetic_angry)
        all_neutral.extend(synthetic_neutral)
        print(f"Added {len(synthetic_angry)} synthetic angry examples")
        print(f"Added {len(synthetic_neutral)} synthetic neutral examples")

    # Add HuggingFace dataset examples
    if include_hf_datasets:
        # Try emotion dataset
        try:
            print("\nLoading emotion dataset...")
            angry, neutral = load_emotion_dataset()
            all_angry.extend(angry)
            all_neutral.extend(neutral)
        except Exception as e:
            print(f"Could not load emotion dataset: {e}")

        # Try GoEmotions dataset
        try:
            print("\nLoading GoEmotions dataset...")
            angry, neutral = load_goemotions_dataset()
            all_angry.extend(angry)
            all_neutral.extend(neutral)
        except Exception as e:
            print(f"Could not load GoEmotions dataset: {e}")

    # Filter and select best examples
    print("\nFiltering and selecting best examples...")
    final_angry, final_neutral = filter_and_select_examples(
        all_angry,
        all_neutral,
        max_angry=max_angry,
        max_neutral=max_neutral
    )

    # Create dataset
    dataset = {
        "angry": final_angry,
        "neutral": final_neutral
    }

    # Save to JSON
    print(f"\nSaving dataset to {output_path}...")
    with open(output_path, 'w') as f:
        json.dump(dataset, f, indent=2)

    print(f"\n{'='*80}")
    print("DATASET CREATION COMPLETE")
    print(f"{'='*80}")
    print(f"Total angry examples: {len(final_angry)}")
    print(f"Total neutral examples: {len(final_neutral)}")
    print(f"Saved to: {output_path}")

    # Show some examples
    print(f"\n{'='*80}")
    print("SAMPLE ANGRY EXAMPLES")
    print(f"{'='*80}")
    for i, example in enumerate(final_angry[:5]):
        print(f"\n{i+1}. {example}")

    print(f"\n{'='*80}")
    print("SAMPLE NEUTRAL EXAMPLES")
    print(f"{'='*80}")
    for i, example in enumerate(final_neutral[:5]):
        print(f"\n{i+1}. {example}")

    return dataset


def main():
    parser = argparse.ArgumentParser(
        description="Create improved emotion dataset from HuggingFace"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="/workspace/basic_science_resampling/emotion_examples.json",
        help="Output path for JSON file"
    )
    parser.add_argument(
        "--max-angry",
        type=int,
        default=100,
        help="Maximum number of angry examples"
    )
    parser.add_argument(
        "--max-neutral",
        type=int,
        default=100,
        help="Maximum number of neutral examples"
    )
    parser.add_argument(
        "--no-hf-datasets",
        action="store_true",
        help="Don't include HuggingFace datasets (synthetic only)"
    )
    parser.add_argument(
        "--no-synthetic",
        action="store_true",
        help="Don't include synthetic examples (HF only)"
    )

    args = parser.parse_args()

    create_improved_dataset(
        output_path=args.output,
        include_hf_datasets=not args.no_hf_datasets,
        include_synthetic=not args.no_synthetic,
        max_angry=args.max_angry,
        max_neutral=args.max_neutral
    )


if __name__ == "__main__":
    main()
