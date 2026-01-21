"""
Script to prepare data for DeepSeek Coder training.

This script helps convert your Python files and prompts into the required format.
"""

import argparse
import json
import pathlib
import random
from typing import Dict, List


def create_dataset_from_pairs(
    prompts_dir: str,
    code_dir: str,
    output_dir: str,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    partial_ratio: float = 0.5,
):
    """
    Create train/val/test split from paired prompt and code files.

    Args:
        prompts_dir: Directory containing .txt prompt files
        code_dir: Directory containing .py code files
        output_dir: Where to save train.jsonl, val.jsonl, test.jsonl
        train_ratio: Fraction for training
        val_ratio: Fraction for validation
        partial_ratio: Default partial code ratio
    """
    prompts_path = pathlib.Path(prompts_dir)
    code_path = pathlib.Path(code_dir)
    output_path = pathlib.Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Find matching pairs
    samples = []

    # Method 1: Match by filename (e.g., sample_001.txt <-> sample_001.py)
    for prompt_file in prompts_path.glob("**/*.txt"):
        # Get corresponding code file
        relative_path = prompt_file.relative_to(prompts_path)
        code_file = code_path / relative_path.with_suffix(".py")

        if code_file.exists():
            # Read prompt
            with open(prompt_file, "r", encoding="utf-8") as f:
                prompt = f.read().strip()

            # Read code
            with open(code_file, "r", encoding="utf-8") as f:
                code = f.read()

            # Create sample
            sample_id = (
                str(relative_path.with_suffix("")).replace("/", "_").replace("\\", "_")
            )
            samples.append(
                {
                    "id": sample_id,
                    "prompt": prompt,
                    "code": code,
                    "partial_ratio": partial_ratio,
                }
            )

    if not samples:
        print(f"No matching pairs found in {prompts_dir} and {code_dir}")
        return

    print(f"Found {len(samples)} prompt-code pairs")

    # Shuffle
    random.seed(42)
    random.shuffle(samples)

    # Split
    n = len(samples)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train_samples = samples[:train_end]
    val_samples = samples[train_end:val_end]
    test_samples = samples[val_end:]

    # Save as JSONL
    def save_jsonl(data: List[Dict], filename: str):
        filepath = output_path / filename
        with open(filepath, "w", encoding="utf-8") as f:
            for item in data:
                f.write(json.dumps(item) + "\n")
        print(f"Saved {len(data)} samples to {filepath}")

    save_jsonl(train_samples, "train.jsonl")
    save_jsonl(val_samples, "val.jsonl")
    save_jsonl(test_samples, "test.jsonl")

    # Print statistics
    print("\nDataset Statistics:")
    print(f"  Total samples: {n}")
    print(f"  Training: {len(train_samples)} ({train_ratio*100:.1f}%)")
    print(f"  Validation: {len(val_samples)} ({val_ratio*100:.1f}%)")
    print(f"  Test: {len(test_samples)} ({(1-train_ratio-val_ratio)*100:.1f}%)")


def create_dataset_from_single_json(
    input_json: str,
    output_dir: str,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
):
    """
    Create splits from a single JSON file.

    Expected format:
    [
        {"id": "...", "prompt": "...", "code": "..."},
        ...
    ]
    """
    input_path = pathlib.Path(input_json)
    output_path = pathlib.Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load JSON
    with open(input_path, "r", encoding="utf-8") as f:
        samples = json.load(f)

    if not isinstance(samples, list):
        print("Error: JSON file must contain a list of samples")
        return

    print(f"Loaded {len(samples)} samples from {input_json}")

    # Shuffle
    random.seed(42)
    random.shuffle(samples)

    # Split
    n = len(samples)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train_samples = samples[:train_end]
    val_samples = samples[train_end:val_end]
    test_samples = samples[val_end:]

    # Save
    def save_jsonl(data: List[Dict], filename: str):
        filepath = output_path / filename
        with open(filepath, "w", encoding="utf-8") as f:
            for item in data:
                f.write(json.dumps(item) + "\n")
        print(f"Saved {len(data)} samples to {filepath}")

    save_jsonl(train_samples, "train.jsonl")
    save_jsonl(val_samples, "val.jsonl")
    save_jsonl(test_samples, "test.jsonl")


def create_sample_data(output_dir: str, num_samples: int = 100):
    """
    Create sample dataset for testing.
    """
    output_path = pathlib.Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Sample prompts and code
    templates = [
        {
            "prompt": "Create a function that calculates factorial of a number",
            "code": """def factorial(n):
    if n == 0 or n == 1:
        return 1
    return n * factorial(n - 1)""",
        },
        {
            "prompt": "Implement a function to check if a string is a palindrome",
            "code": """def is_palindrome(s):
    s = s.lower().replace(' ', '')
    return s == s[::-1]""",
        },
        {
            "prompt": "Create a function to find the maximum element in a list",
            "code": """def find_max(lst):
    if not lst:
        return None
    max_val = lst[0]
    for val in lst[1:]:
        if val > max_val:
            max_val = val
    return max_val""",
        },
        {
            "prompt": "Write a function to reverse a linked list",
            "code": """def reverse_linked_list(head):
    prev = None
    current = head
    while current:
        next_node = current.next
        current.next = prev
        prev = current
        current = next_node
    return prev""",
        },
    ]

    # Generate samples
    samples = []
    for i in range(num_samples):
        template = templates[i % len(templates)]
        samples.append(
            {
                "id": f"sample_{i:04d}",
                "prompt": template["prompt"],
                "code": template["code"],
                "partial_ratio": 0.5,
            }
        )

    # Split
    random.seed(42)
    random.shuffle(samples)

    n = len(samples)
    train_end = int(n * 0.8)
    val_end = int(n * 0.9)

    train_samples = samples[:train_end]
    val_samples = samples[train_end:val_end]
    test_samples = samples[val_end:]

    # Save
    def save_jsonl(data: List[Dict], filename: str):
        filepath = output_path / filename
        with open(filepath, "w", encoding="utf-8") as f:
            for item in data:
                f.write(json.dumps(item) + "\n")
        print(f"Saved {len(data)} samples to {filepath}")

    save_jsonl(train_samples, "train.jsonl")
    save_jsonl(val_samples, "val.jsonl")
    save_jsonl(test_samples, "test.jsonl")

    print(f"\nCreated sample dataset in {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare data for DeepSeek Coder training"
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # From paired files
    pairs_parser = subparsers.add_parser(
        "pairs", help="Create dataset from paired files"
    )
    pairs_parser.add_argument(
        "--prompts_dir", required=True, help="Directory with prompt .txt files"
    )
    pairs_parser.add_argument(
        "--code_dir", required=True, help="Directory with code .py files"
    )
    pairs_parser.add_argument("--output_dir", required=True, help="Output directory")
    pairs_parser.add_argument(
        "--train_ratio", type=float, default=0.8, help="Training set ratio"
    )
    pairs_parser.add_argument(
        "--val_ratio", type=float, default=0.1, help="Validation set ratio"
    )
    pairs_parser.add_argument(
        "--partial_ratio", type=float, default=0.5, help="Partial code ratio"
    )

    # From single JSON
    json_parser = subparsers.add_parser(
        "json", help="Create dataset from single JSON file"
    )
    json_parser.add_argument("--input_json", required=True, help="Input JSON file")
    json_parser.add_argument("--output_dir", required=True, help="Output directory")
    json_parser.add_argument(
        "--train_ratio", type=float, default=0.8, help="Training set ratio"
    )
    json_parser.add_argument(
        "--val_ratio", type=float, default=0.1, help="Validation set ratio"
    )

    # Create sample data
    sample_parser = subparsers.add_parser(
        "sample", help="Create sample dataset for testing"
    )
    sample_parser.add_argument(
        "--output_dir", default="./sample_data", help="Output directory"
    )
    sample_parser.add_argument(
        "--num_samples", type=int, default=100, help="Number of samples"
    )

    args = parser.parse_args()

    if args.command == "pairs":
        create_dataset_from_pairs(
            prompts_dir=args.prompts_dir,
            code_dir=args.code_dir,
            output_dir=args.output_dir,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            partial_ratio=args.partial_ratio,
        )
    elif args.command == "json":
        create_dataset_from_single_json(
            input_json=args.input_json,
            output_dir=args.output_dir,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
        )
    elif args.command == "sample":
        create_sample_data(
            output_dir=args.output_dir,
            num_samples=args.num_samples,
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
