"""
Dataset for training DeepSeek Coder with prompt + partial code → complete code.

This dataset handles pairs of:
- Input: Natural language prompt + partial Python file
- Output: Complete Python file (including the partial part)
"""

import json
import os
import pathlib
import random
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoTokenizer

# Disable tokenizer parallelism warning when using with DataLoader workers
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class PromptCompletionDataset(Dataset):
    def __init__(
        self,
        config,
        data_path: Optional[str] = None,
        tokenize: bool = True,
        split: str = "train",
    ):
        """
        Initialize the dataset.

        Args:
            config: Configuration object
            data_path: Path to JSON file or directory with data
            tokenize: Whether to pre-tokenize the data
            split: Dataset split ('train', 'val', 'test')
        """
        self.config = config
        self.tokenize = tokenize
        self.split = split

        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.cadcoder.model_name, trust_remote_code=True
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"

        # Set random seeds for reproducibility
        torch.manual_seed(config.data.seed)
        np.random.seed(config.data.seed)
        random.seed(config.data.seed)

        # Load data
        self.data = self._load_data(data_path or config.data.root_dir)

        # Optionally pre-tokenize
        if self.tokenize:
            self.max_length = config.data.max_total_len
            self._tokenize_data()

    def _load_data(self, data_path: str) -> List[Dict]:
        """Load data from JSON file, JSONL, directory, or CSV with Python files."""
        data_path = pathlib.Path(data_path)
        samples = []

        # Check for CSV-based format (prompts.csv + python files)
        csv_path = data_path / "prompt" / "prompts.csv" if data_path.is_dir() else None
        if csv_path and csv_path.exists():
            # Load from CSV format
            samples = self._load_from_csv(data_path)

        elif data_path.is_file() and data_path.suffix == ".json":
            # Single JSON file
            with open(data_path, "r") as f:
                raw_data = json.load(f)
                for item in raw_data:
                    samples.append(self._process_sample(item))

        elif data_path.is_file() and data_path.suffix == ".jsonl":
            # JSONL file (one JSON per line)
            with open(data_path, "r") as f:
                for line in f:
                    item = json.loads(line)
                    samples.append(self._process_sample(item))

        elif data_path.is_dir():
            # Directory with JSON files
            for json_file in data_path.glob("**/*.json"):
                with open(json_file, "r") as f:
                    raw_data = json.load(f)
                    if isinstance(raw_data, list):
                        for item in raw_data:
                            samples.append(self._process_sample(item))
                    else:
                        samples.append(self._process_sample(raw_data))

        else:
            raise ValueError(f"Invalid data path: {data_path}")

        print(f"Loaded {len(samples)} samples for {self.split} split")
        return samples

    def _load_from_csv(self, base_path: pathlib.Path) -> List[Dict]:
        """
        Load data from CSV file with prompts and corresponding Python files.

        Expected structure:
        base_path/
            prompts/
                prompts.csv  (columns: 'file', 'prompt')
            python/
                file1.py
                file2.py
                ...

        Args:
            base_path: Base directory containing prompts/ and python/ subdirectories

        Returns:
            List of processed samples
        """
        import csv

        csv_path = base_path / "prompt" / "prompts.csv"
        python_dir = base_path / "python"

        if not csv_path.exists():
            raise ValueError(f"CSV file not found: {csv_path}")
        if not python_dir.exists():
            raise ValueError(f"Python directory not found: {python_dir}")

        samples = []

        # Read CSV file
        with open(csv_path, "r", encoding="utf-8") as f:
            # Try to detect the delimiter
            first_line = f.readline()
            f.seek(0)

            # Common delimiters to try
            delimiter = ","
            if "\t" in first_line:
                delimiter = "\t"
            elif ";" in first_line:
                delimiter = ";"

            print(f"Using delimiter: {repr(delimiter)}")
            print(f"CSV header line: {first_line.strip()}")

            reader = csv.DictReader(f, delimiter=delimiter)

            # Print fieldnames to debug
            if reader.fieldnames:
                print(f"CSV columns detected: {reader.fieldnames}")

            for row_idx, row in enumerate(reader):
                # Handle case where row is a string or dict
                if isinstance(row, str):
                    # If row is a string, something went wrong
                    print(
                        f"Warning: CSV row {row_idx} is a string, not a dict: {row[:100]}"
                    )
                    continue

                # Debug first row
                if row_idx == 0:
                    print(f"First row keys: {list(row.keys())}")
                    print(
                        f"First row values: {list(row.values())[:2]}"
                    )  # First 2 values

                filename = row.get("file", "").strip()
                prompt = row.get("prompt", "").strip()

                if not filename or not prompt:
                    if row_idx < 3:  # Only warn for first few rows
                        print(f"Warning: Row {row_idx} missing file or prompt")
                    continue

                # Find corresponding Python file
                python_file = python_dir / filename

                if not python_file.exists():
                    print(f"Warning: Python file not found: {python_file}")
                    continue

                # Read Python code
                try:
                    with open(python_file, "r", encoding="utf-8") as code_file:
                        code = code_file.read()
                except Exception as e:
                    print(f"Error reading {python_file}: {e}")
                    continue

                # Create sample
                sample_id = (
                    filename.replace(".py", "").replace("/", "_").replace("\\", "_")
                )
                sample = {
                    "id": sample_id,
                    "prompt": prompt,
                    "code": code,
                    "partial_ratio": self.config.data.partial_ratio,
                }

                samples.append(self._process_sample(sample))

        print(f"Loaded {len(samples)} samples from CSV format")
        return samples

    def _process_sample(self, item: Dict) -> Dict:
        """Process a single sample from raw data."""
        prompt = item.get("prompt", "")
        code = item.get("code", "")
        partial_ratio = item.get("partial_ratio", self.config.data.partial_ratio)

        # Generate partial code by taking first N% of lines or characters
        partial_code = self._create_partial_code(code, partial_ratio)

        return {
            "id": item.get("id", ""),
            "prompt": prompt,
            "partial_code": partial_code,
            "full_code": code,
        }

    def _create_partial_code(self, code: str, ratio: float) -> str:
        """
        Create partial code from full code.

        Args:
            code: Full Python code
            ratio: Ratio of code to include (0.0 to 1.0)

        Returns:
            Partial code string
        """
        if ratio <= 0:
            return ""
        if ratio >= 1.0:
            return code

        # Split by lines and take first N%
        lines = code.split("\n")
        num_lines = max(1, int(len(lines) * ratio))
        return "\n".join(lines[:num_lines])

    def _tokenize_data(self):
        """Pre-tokenize all samples and filter out those that are too long."""
        print(f"Pre-tokenizing {self.split} samples...")
        filtered_data = []
        truncated_count = 0

        for sample in tqdm(self.data, desc=f"Tokenizing {self.split}"):
            # Format the input (prompt + partial code)
            input_text = self._format_input(sample["prompt"], sample["partial_code"])
            target_text = sample["full_code"]

            # Check length
            input_len = len(self.tokenizer.encode(input_text))
            target_len = len(self.tokenizer.encode(target_text))
            total_len = input_len + target_len

            if total_len > self.config.data.max_total_len:
                truncated_count += 1
                continue

            # Tokenize
            tokenized_input = self.tokenizer(
                input_text,
                return_tensors="pt",
                add_special_tokens=True,
                truncation=False,
            )

            tokenized_target = self.tokenizer(
                target_text,
                return_tensors="pt",
                add_special_tokens=False,  # Don't add BOS again
                truncation=False,
            )

            sample["tokenized_input"] = tokenized_input
            sample["tokenized_target"] = tokenized_target
            sample["input_length"] = input_len
            sample["target_length"] = target_len

            filtered_data.append(sample)

        print(f"Filtered out {truncated_count} samples that were too long")
        print(f"Remaining samples: {len(filtered_data)}")
        self.data = filtered_data

    def _format_input(self, prompt: str, partial_code: str) -> str:
        """
        Format the input text for the model.

        This uses the DeepSeek Coder instruct format:
        ### Instruction:
        {prompt}

        ### Code:
        {partial_code}

        ### Response:
        """
        if partial_code:
            formatted = f"""### Instruction:
{prompt}

### Code:
{partial_code}

### Response:
"""
        else:
            formatted = f"""### Instruction:
{prompt}

### Response:
"""
        return formatted

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict:
        sample = self.data[idx]

        if self.tokenize:
            # Return pre-tokenized data
            return {
                "id": sample["id"],
                "input_ids": sample["tokenized_input"]["input_ids"].squeeze(0),
                "target_ids": sample["tokenized_target"]["input_ids"].squeeze(0),
                "input_length": sample["input_length"],
                "target_length": sample["target_length"],
            }
        else:
            # Return raw text (tokenize on-the-fly in collate_fn)
            input_text = self._format_input(sample["prompt"], sample["partial_code"])
            return {
                "id": sample["id"],
                "input_text": input_text,
                "target_text": sample["full_code"],
                "prompt": sample["prompt"],
                "partial_code": sample["partial_code"],
            }

    def collate_fn(self, batch: List[Dict]) -> Dict:
        """
        Collate function for DataLoader.
        Handles padding and creates attention masks.
        """
        if self.tokenize:
            # Pre-tokenized case
            input_ids_list = [item["input_ids"] for item in batch]
            target_ids_list = [item["target_ids"] for item in batch]

            # Combine input and target for training
            combined_ids = []
            for inp, tgt in zip(input_ids_list, target_ids_list):
                combined = torch.cat([inp, tgt], dim=0)
                combined_ids.append(combined)

            # Pad sequences
            input_ids = torch.nn.utils.rnn.pad_sequence(
                combined_ids,
                batch_first=True,
                padding_value=self.tokenizer.pad_token_id,
            )

            # Create attention mask
            attention_mask = (input_ids != self.tokenizer.pad_token_id).long()

            # Create labels (mask out the input part, only compute loss on target)
            labels = input_ids.clone()
            for i, item in enumerate(batch):
                # Mask the input portion with -100 (ignored in loss)
                labels[i, : item["input_length"]] = -100

            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
                "ids": [item["id"] for item in batch],
            }

        else:
            # On-the-fly tokenization
            input_texts = [item["input_text"] for item in batch]
            target_texts = [item["target_text"] for item in batch]

            # Tokenize inputs
            input_encoded = self.tokenizer(
                input_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.config.data.max_total_len,
                add_special_tokens=True,
            )

            # Tokenize targets
            target_encoded = self.tokenizer(
                target_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.config.data.max_total_len,
                add_special_tokens=False,
            )

            # Combine for training
            batch_size = len(batch)
            max_input_len = input_encoded["input_ids"].shape[1]
            max_target_len = target_encoded["input_ids"].shape[1]
            max_total_len = max_input_len + max_target_len

            input_ids = torch.full(
                (batch_size, max_total_len),
                self.tokenizer.pad_token_id,
                dtype=torch.long,
            )
            labels = torch.full(
                (batch_size, max_total_len),
                -100,
                dtype=torch.long,
            )

            for i in range(batch_size):
                # Copy input
                inp_len = (input_encoded["attention_mask"][i] == 1).sum()
                input_ids[i, :inp_len] = input_encoded["input_ids"][i, :inp_len]

                # Copy target
                tgt_len = (target_encoded["attention_mask"][i] == 1).sum()
                input_ids[i, inp_len : inp_len + tgt_len] = target_encoded["input_ids"][
                    i, :tgt_len
                ]
                labels[i, inp_len : inp_len + tgt_len] = target_encoded["input_ids"][
                    i, :tgt_len
                ]

            attention_mask = (input_ids != self.tokenizer.pad_token_id).long()

            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
                "ids": [item["id"] for item in batch],
                "prompts": [item["prompt"] for item in batch],
            }

    @staticmethod
    def create_splits(
        config, data_path: Optional[str] = None
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Create train, validation, and test dataloaders.

        Args:
            config: Configuration object
            data_path: Path to data directory or file

        Returns:
            Tuple of (train_loader, val_loader, test_loader)
        """
        data_path = data_path or config.data.root_dir
        data_path = pathlib.Path(data_path)

        # Load all data and split
        full_dataset = PromptCompletionDataset(
            config, data_path=data_path, tokenize=True, split="full"
        )

        # Split ratios
        train_ratio = config.data.train_ratio
        val_ratio = config.data.val_ratio
        test_ratio = 1.0 - train_ratio - val_ratio

        total_size = len(full_dataset)
        train_size = int(train_ratio * total_size)
        val_size = int(val_ratio * total_size)
        test_size = total_size - train_size - val_size

        train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
            full_dataset,
            [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(config.data.seed),
        )

        collate_fn = full_dataset.collate_fn

        # Create dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.data.num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.data.num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.data.num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
        )

        return train_loader, val_loader, test_loader
