import pathlib
import random
import re

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm
from transformers import AutoTokenizer


class RapidcadpyDataset(Dataset):
    def __init__(
        self,
        config,
        tokenize=True,
    ):
        """
        Dataset for loading Python CAD code without text prompts.

        Args:
            config: Configuration object with dataset parameters
            tokenize: Whether to tokenize the code during initialization
        """
        max_samples = config.data.dataset_size

        self.config = config
        self.tokenize = tokenize

        # Initialize tokenizer
        self.code_tokenizer = AutoTokenizer.from_pretrained(
            config.cadcoder.model_name, trust_remote_code=True
        )
        self.code_tokenizer.pad_token = self.code_tokenizer.eos_token
        self.code_tokenizer.padding_side = "right"

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

        # Collect all code samples
        self.data = []
        count = 0

        if cadquery_df is not None:
            for idx, row in cadquery_df.iterrows():
                code = row["cadquery"]
                deepcad_id = row["deepcad_id"]

                # Round floats in code for consistency
                code = RapidcadpyDataset.round_floats_in_code(code, digits=2)

                # Skip if code is too long (only if not tokenizing during init)
                if not self.tokenize:
                    if (
                        len(self.code_tokenizer.encode(code))
                        > self.config.data.max_total_len
                    ):
                        continue

                sample = {
                    "code": code,
                    "id": deepcad_id,
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
            sample["original_token_length"] = original_length

            filtered_data.append(sample)

        self.data = filtered_data

        # Report truncation statistics
        if truncated_count > 0:
            truncation_rate = ((truncated_count + 1) / (len(self.data) + 1)) * 100
            print(
                f"{truncated_count}/{len(self.data)} samples ({truncation_rate:.1f}%) were truncated"
            )

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
        dataset = RapidcadpyDataset(config, tokenize=tokenize)

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
        ids = [item["id"] for item in batch]

        tokenized_code = self.code_tokenizer(
            codes,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.config.data.max_total_len,
            add_special_tokens=True,
        )

        return {
            "input_ids": tokenized_code["input_ids"],
            "attention_mask": tokenized_code["attention_mask"],
            "code": codes,
            "ids": ids,
        }

    def collate_with_tokenization(self, batch):
        """
        Custom collate function for pre-tokenized data.
        """
        codes = [item["code"] for item in batch]
        ids = [item["id"] for item in batch]

        input_ids = torch.stack([item["input_ids"] for item in batch])
        attention_mask = torch.stack([item["attention_mask"] for item in batch])
        labels = torch.stack([item["labels"] for item in batch])

        result = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "code": codes,
            "ids": ids,
            "labels": labels,
        }

        return result
