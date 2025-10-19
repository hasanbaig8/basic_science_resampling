"""
Load StrategyQA dataset from HuggingFace and save to JSON.

Loads from https://huggingface.co/datasets/ChilleD/StrategyQA,
taking the question and answer columns and saves to a json file.
"""

import json
from datasets import load_dataset


def main():
    """Load StrategyQA dataset and save question-answer pairs to JSON."""
    print("Loading StrategyQA dataset from HuggingFace...")

    # Load the dataset
    dataset = load_dataset("ChilleD/StrategyQA")

    # Extract train split (or use 'test' if preferred)
    train_data = dataset['train']

    print(f"Loaded {len(train_data)} examples")

    # Extract question and answer columns with question_id
    qa_pairs = []
    for idx, item in enumerate(train_data):
        qa_pairs.append({
            "question_id": idx,
            "question": item["question"],
            "answer": item["answer"]
        })

    # Save to JSON file
    output_file = "data/strategyqa_data.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(qa_pairs, f, indent=2, ensure_ascii=False)

    print(f"Saved {len(qa_pairs)} question-answer pairs to {output_file}")

    # Print a sample
    if qa_pairs:
        print("\nSample entry:")
        print(json.dumps(qa_pairs[0], indent=2))


if __name__ == "__main__":
    main()