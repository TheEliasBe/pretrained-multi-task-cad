import json
import os
import pathlib
import random
import re
import statistics

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm
from transformers import AutoTokenizer


class TextToPyDataset(Dataset):
    def __init__(
        self,
        config,
        tokenize=True,
    ):
        """
        Dataset for loading text prompts and corresponding Python code.

        Args:
            config: Configuration object with dataset parameters
            text_dir: Directory containing JSON files with text prompts
            code_dir: Directory containing Python code files (can have subdirectories)
            tokenize: Whether to tokenize the code during initialization
        """
        text_dir = pathlib.Path(config.data.root_dir) / "omni_cad" / "Omni-CAD" / "txt"
        max_samples = config.data.dataset_size

        self.config = config
        self.tokenize = tokenize

        # Initialize tokenizers
        self.code_tokenizer = AutoTokenizer.from_pretrained(
            config.cadcoder.model_name, trust_remote_code=True
        )
        self.code_tokenizer.pad_token = self.code_tokenizer.eos_token
        self.code_tokenizer.padding_side = "right"
        self.text_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

        if self.tokenize:
            self.max_length = config.data.max_total_len

        # Set random seeds
        torch.manual_seed(config.data.seed)
        np.random.seed(config.data.seed)
        random.seed(config.data.seed)

        # Load all parquet files in data_dir and concat, limit to max_samples
        cadquery_df = None
        if config.data.use_cad_query_code:
            data_dir = pathlib.Path(config.data.root_dir) / "GenCAD-Code" / "data"
            parquet_files = sorted(data_dir.glob("*.parquet"))
            dfs = []
            total_rows = 0
            for pf in parquet_files:
                df = pd.read_parquet(pf)
                if max_samples and total_rows + len(df) > max_samples:
                    df = df.iloc[: max_samples - total_rows]
                dfs.append(df)
                total_rows += len(df)
                if max_samples and total_rows >= max_samples:
                    break
            cadquery_df = pd.concat(dfs, ignore_index=True)

        # Collect all text prompt files
        self.text_files = []
        for root, _, files in os.walk(text_dir):
            for file in files:
                if file.endswith(".json"):
                    self.text_files.append(os.path.join(root, file))

        # Create mapping from 8-digit prefix to text prompts
        self.text_map = {}
        for text_file in self.text_files:
            with open(text_file, "r") as f:
                data = json.load(f)
                for item in data:
                    # Extract 8-digit prefix from ID (first 8 characters)
                    prefix = item["id"].split("/")[-1][:8]
                    self.text_map[prefix] = item["text caption"]

        # Collect all code files recursively and match with text prompts
        self.data = []
        count = 0

        for id in self.text_map.keys():
            deepcad_id = f"{id[:4]}/{id}"
            matches = cadquery_df[cadquery_df["deepcad_id"] == deepcad_id][
                "cadquery"
            ].values
            if len(matches) == 0:
                continue
            code = matches[0]
            code = TextToPyDataset.round_floats_in_code(code, digits=2)

            # Skip if code is too long (only if not tokenizing during init)
            if not self.tokenize:
                if (
                    len(self.code_tokenizer.encode(code))
                    > self.config.data.max_total_len
                ):
                    continue

            sample = {
                "text": self.text_map[id],
                "code": code,
                "id": id,
            }

            self.data.append(sample)
            count += 1
            if max_samples and count >= max_samples:
                break

        # Tokenize if requested
        if self.tokenize:
            self._tokenize_data()

    def _tokenize_data(self):
        """Tokenize all code samples and filter out those that are too long"""
        print("Pre-tokenizing code samples...")
        truncated_count = 0
        filtered_data = []

        for sample in tqdm(self.data, desc="Tokenizing"):
            original_length = len(self.code_tokenizer.encode(sample["code"]))

            # Skip if this sample would be truncated
            if original_length > self.max_length:
                truncated_count += 1
                continue

            tokenized_code = self.code_tokenizer(
                sample["code"],
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
                add_special_tokens=True,
            )

            tokenized_text = self.text_tokenizer(
                sample["text"],
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=512,
            )

            # add -100 labels for padding tokens so they are except from loss computation
            pad_labels = torch.full(
                (1, self.config.cadcoder.num_virtual_tokens),
                -100,
            )
            labels = torch.cat([pad_labels, tokenized_code["input_ids"]], dim=1)
            first_eos = (labels == self.code_tokenizer.eos_token_id).cumsum(dim=1)
            labels[first_eos > 1] = -100

            sample["labels"] = labels.squeeze(0)

            sample["input_ids"] = tokenized_code["input_ids"].squeeze(0)
            sample["attention_mask"] = (
                tokenized_code["attention_mask"].squeeze(0).bool()
            )
            sample["text_input_ids"] = tokenized_text["input_ids"].squeeze(0)
            sample["text_attention_mask"] = tokenized_text["attention_mask"].squeeze(0)
            sample["original_token_length"] = original_length

            filtered_data.append(sample)

        self.data = filtered_data

        # Report truncation statistics
        if truncated_count > 0:
            truncation_rate = ((truncated_count + 1) / (len(self.data) + 1)) * 100
            print(
                f"{truncated_count}/{len(self.data)} samples ({truncation_rate:.1f}%) were truncated"
            )

    def _analyze_token_statistics(self):
        """Analyze token count statistics to help optimize sequence length"""
        print("\n📊 Analyzing token count statistics...")

        token_lengths = []

        # Sample a subset for analysis if dataset is large
        sample_size = min(len(self.data), 1000)
        sample_indices = np.random.choice(len(self.data), sample_size, replace=False)

        for idx in tqdm(sample_indices, desc="Analyzing tokens"):
            sample = self.data[idx]
            tokens = self.code_tokenizer.encode(sample["code"])
            token_lengths.append(len(tokens))

        # Calculate statistics
        mean_length = statistics.mean(token_lengths)
        median_length = statistics.median(token_lengths)
        std_length = statistics.stdev(token_lengths) if len(token_lengths) > 1 else 0
        min_length = min(token_lengths)
        max_length = max(token_lengths)

        # Calculate percentiles
        percentiles = [50, 75, 90, 95, 99]
        percentile_values = [np.percentile(token_lengths, p) for p in percentiles]

        # Print comprehensive statistics
        print(f"\n🔢 Token Length Statistics (n={sample_size}):")
        print(f"   Mean:     {mean_length:.1f} tokens")
        print(f"   Median:   {median_length:.1f} tokens")
        print(f"   Std Dev:  {std_length:.1f} tokens")
        print(f"   Min:      {min_length} tokens")
        print(f"   Max:      {max_length} tokens")
        print("\n📈 Percentiles:")
        for p, v in zip(percentiles, percentile_values):
            print(f"   {p}th:     {v:.0f} tokens")

        # Recommendations
        print("\n💡 Sequence Length Recommendations:")
        print(f"   Conservative (covers 90%): {percentile_values[2]:.0f}")
        print(f"   Balanced (covers 95%):     {percentile_values[3]:.0f}")
        print(f"   Aggressive (covers 99%):   {percentile_values[4]:.0f}")
        print(f"   Current setting:           {self.max_length}")

        # Memory usage estimates
        if hasattr(self.config, "decoder") and hasattr(
            self.config.decoder, "dim_hidden"
        ):
            hidden_size = self.config.decoder.dim_hidden

            print("\n💾 Memory Impact Estimates (per sample):")
            for desc, length in [
                ("90% coverage", percentile_values[2]),
                ("95% coverage", percentile_values[3]),
                ("99% coverage", percentile_values[4]),
                ("Current", self.max_length),
            ]:
                # Rough memory estimate: input_ids + attention_mask + embeddings
                memory_mb = (
                    length * 4  # input_ids (int32)
                    + length * 4  # attention_mask (int32)
                    + length * hidden_size * 2  # embeddings (bfloat16)
                ) / (1024 * 1024)
                print(f"   {desc:<15}: ~{memory_mb:.1f} MB per sample")

        # Store statistics for later use
        self.token_stats = {
            "mean": mean_length,
            "median": median_length,
            "std": std_length,
            "min": min_length,
            "max": max_length,
            "percentiles": dict(zip(percentiles, percentile_values)),
            "sample_size": sample_size,
        }

    def get_token_statistics(self):
        """Return token statistics for external use"""
        return getattr(self, "token_stats", None)

    def get_truncation_report(self):
        """Generate a detailed truncation report"""
        if (
            not hasattr(self, "data")
            or not self.data
            or "original_token_length" not in self.data[0]
        ):
            return "Truncation analysis not available"

        original_lengths = [sample["original_token_length"] for sample in self.data]
        truncated = [length > self.max_length for length in original_lengths]

        truncated_count = sum(truncated)
        truncation_rate = ((truncated_count + 1) / (len(self.data) + 1)) * 100

        if truncated_count > 0:
            truncated_lengths = [
                length
                for length, is_trunc in zip(original_lengths, truncated)
                if is_trunc
            ]
            avg_truncated_length = statistics.mean(truncated_lengths)
            avg_lost_tokens = avg_truncated_length - self.max_length

            return {
                "truncated_samples": truncated_count,
                "total_samples": len(self.data),
                "truncation_rate": truncation_rate,
                "avg_original_length_truncated": avg_truncated_length,
                "avg_tokens_lost": avg_lost_tokens,
                "max_length_setting": self.max_length,
            }
        else:
            return {
                "truncated_samples": 0,
                "total_samples": len(self.data),
                "truncation_rate": 0.0,
                "max_length_setting": self.max_length,
            }

    @staticmethod
    def round_floats_in_code(code: str, digits: int = 6) -> str:
        # Match floats like 0.12345678, -1.2345, etc.
        float_pattern = re.compile(r"-?\d+\.\d+")

        def round_match(match):
            num = float(match.group())
            return str(round(num, digits))

        return float_pattern.sub(round_match, code)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        return sample

    @staticmethod
    def create_splits(config, tokenize=True):
        # Create one dataset instance
        dataset = TextToPyDataset(config, tokenize=tokenize)

        train_ds, val_ds, test_ds = random_split(dataset, [0.8, 0.1, 0.1])

        # Choose appropriate collate function and batch size
        if tokenize:
            collate_fn = dataset.collate_with_tokenization
        else:
            collate_fn = dataset.collate

        train_dl = DataLoader(
            train_ds,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.data.num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
            drop_last=True,
        )
        val_dl = DataLoader(
            val_ds,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.data.num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
            drop_last=True,
        )
        test_dl = DataLoader(
            test_ds,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.data.num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
            drop_last=True,
        )
        return train_dl, val_dl, test_dl

    def collate(self, batch):
        """
        Custom collate function to handle batching without pre-tokenization.
        """
        self.code_tokenizer.pad_token = self.code_tokenizer.eos_token

        codes = [item["code"] for item in batch]
        texts = [item["text"] for item in batch]
        ids = [item["id"] for item in batch]

        tokenized_code = self.code_tokenizer(
            codes,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.config.data.max_total_len,
            add_special_tokens=True,
        )

        tokenized_text = self.text_tokenizer(
            texts,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=512,
        )

        return {
            "text": tokenized_text["input_ids"],
            "text_attention_mask": tokenized_text["attention_mask"],
            "input_ids": tokenized_code["input_ids"],
            "attention_mask": tokenized_code["attention_mask"],
            "code_path": code_paths,
            "id": ids,
        }

    def collate_with_tokenization(self, batch):
        """
        Custom collate function for pre-tokenized data.
        """
        codes = [item["code"] for item in batch]
        ids = [item["id"] for item in batch]

        input_ids = torch.stack([item["input_ids"] for item in batch])
        attention_mask = torch.stack([item["attention_mask"] for item in batch])
        text_input_ids = torch.stack([item["text_input_ids"] for item in batch])
        text_attention_mask = torch.stack(
            [item["text_attention_mask"] for item in batch]
        )
        labels = torch.stack([item["labels"] for item in batch])

        result = {
            "text": text_input_ids,
            "text_attention_mask": text_attention_mask,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "code": codes,
            "ids": ids,
            "labels": labels,
        }

        return result
