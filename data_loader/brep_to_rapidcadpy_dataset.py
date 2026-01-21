import pathlib
import random
import re
import statistics

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm


class BrepToPyDataset(Dataset):
    def __init__(
        self,
        config,
        tokenize=True,
    ):
        """
        Dataset for loading B-Rep graphs and corresponding Python code.

        Args:
            config: Configuration object with dataset parameters
            text_csv: Path to the CSV file with text prompts and IDs
            code_dir: Directory containing Python code files organized by subfolders
            graph_root_dir: Directory containing B-Rep graph files (.bin format)
            tokenize: Whether to tokenize the code during initialization
        """
        graph_root_dir = pathlib.Path(config.data.root_dir) / "dgl_graphs"
        max_samples = config.data.dataset_size

        self.config = config
        self.tokenize = tokenize

        # Initialize tokenizer if needed
        if self.tokenize:
            from transformers import AutoTokenizer

            model_name = config.cadcoder.model_name
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name, trust_remote_code=True
            )
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.padding_side = "right"
            self.max_length = config.data.max_total_len

        torch.manual_seed(config.data.seed)
        np.random.seed(config.data.seed)
        random.seed(config.data.seed)

        # Load all parquet files in data_dir and concat
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
        self.data = []
        count = 0

        # Iterate through graph folders (e.g., 0000, 0003, etc.)
        graph_folders = sorted([d for d in graph_root_dir.iterdir() if d.is_dir()])

        for graph_folder in tqdm(graph_folders, desc="Processing Graphs"):
            folder_name = graph_folder.name  # e.g., "0000"

            # Get all .bin files in this folder
            graph_files = list(graph_folder.rglob("*.bin"))

            for graph_file in graph_files:
                file_id = graph_file.stem  # e.g., "00000007" from "00000007.bin"
                # Load and validate B-Rep graph
                dgl_graph = dgl.load_graphs(str(graph_file))[0][0]

                # Filter out graphs that are too large
                if (
                    dgl_graph.num_nodes() > self.config.brep_encoder.max_nodes
                    or dgl_graph.num_edges() > self.config.brep_encoder.max_edges
                ):
                    continue

                # Ensure graph data is in correct format
                dgl_graph.ndata["x"] = dgl_graph.ndata["x"].bfloat16()
                dgl_graph.edata["x"] = dgl_graph.edata["x"].bfloat16()

                deepcad_id = f"{folder_name}/{file_id}"
                matches = cadquery_df[cadquery_df["deepcad_id"] == deepcad_id][
                    "cadquery"
                ].values
                if len(matches) == 0:
                    continue
                code = matches[0]
                code = BrepToPyDataset.round_floats_in_code(code, digits=2)

                sample = {
                    "graph": dgl_graph,
                    "code": code,
                    "graph_path": str(graph_file),
                    "id": f"{folder_name}/{file_id}",
                }

                self.data.append(sample)
                count += 1
                if max_samples and count >= max_samples:
                    break
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
            original_length = len(self.tokenizer.encode(sample["code"]))

            # Skip if this sample would be truncated
            if original_length > self.max_length:
                truncated_count += 1
                continue

            tokenized = self.tokenizer(
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
            labels = torch.cat([pad_labels, tokenized["input_ids"]], dim=1)
            first_eos = (labels == self.tokenizer.eos_token_id).cumsum(dim=1)
            labels[first_eos > 1] = -100

            sample["input_ids"] = tokenized["input_ids"].squeeze(0)
            sample["attention_mask"] = tokenized["attention_mask"].squeeze(0).bool()
            sample["labels"] = labels.squeeze(0)
            sample["original_token_length"] = original_length

            filtered_data.append(sample)

        self.data = filtered_data  # replace dataset with filtered list

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
            tokens = self.tokenizer.encode(sample["code"])
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
    def collate(batch):
        """
        Custom collate function to handle batching of graphs and code.
        """
        graphs = [item["graph"] for item in batch]
        codes = [item["code"] for item in batch]
        graph_paths = [item["graph_path"] for item in batch]
        ids = [item["id"] for item in batch]

        # Batch the DGL graphs
        batched_graph = dgl.batch(graphs)

        return {
            "graph": batched_graph,
            "code": codes,
            "graph_path": graph_paths,
            "id": ids,
        }

    def collate_with_tokenization(self, batch):
        """
        Custom collate function for tokenized data.
        """
        graphs = [item["graph"] for item in batch]
        codes = [item["code"] for item in batch]
        graph_paths = [item["graph_path"] for item in batch]
        ids = [item["id"] for item in batch]

        input_ids = torch.stack([item["input_ids"] for item in batch])
        labels = torch.stack([item["labels"] for item in batch])

        raw_attention_mask = torch.stack(
            [item["attention_mask"] for item in batch]
        )  # shape [B, L]

        batched_graph = dgl.batch(graphs)

        B, L = raw_attention_mask.shape
        prefix_len = self.config.cadcoder.num_virtual_tokens
        total_len = prefix_len + L

        device = raw_attention_mask.device
        dtype = raw_attention_mask.dtype

        # Full attention mask [B, T, T] initialized to zeros
        full_attention_mask = torch.zeros(
            (B, total_len, total_len), dtype=dtype, device=device
        )

        # Step 1: Allow prefix tokens to attend to everything
        full_attention_mask[:, :, :prefix_len] = 1

        # Step 2: Causal mask for code tokens (after prefix)
        causal_mask = torch.tril(torch.ones((L, L), dtype=dtype, device=device))
        full_attention_mask[:, prefix_len:, prefix_len:] = causal_mask

        return {
            "input_ids": input_ids,
            "attention_mask": raw_attention_mask,
            "labels": labels,
            "graphs": batched_graph,
            "code": codes,
            "graph_path": graph_paths,
            "ids": ids,
        }

    @staticmethod
    def create_splits(config, tokenize=True):
        # Create one dataset instance
        dataset = BrepToPyDataset(config, tokenize=tokenize)

        # Print final recommendations if tokenization was used
        if tokenize:
            stats = dataset.get_token_statistics()
            if stats:
                print("\n🎯 Final Configuration Recommendations:")
                print(
                    f"   For 90% coverage: max_code_length = {int(stats['percentiles'][90])}"
                )
                print(
                    f"   For 95% coverage: max_code_length = {int(stats['percentiles'][95])}"
                )
                print(
                    f"   For 99% coverage: max_code_length = {int(stats['percentiles'][99])}"
                )

        # Split dataset
        if tokenize:
            train_ds, val_ds, test_ds = random_split(dataset, [0.8, 0.1, 0.1])
        else:
            train_size = int(0.9 * len(dataset))
            val_size = int(0.05 * len(dataset))
            test_size = len(dataset) - train_size - val_size
            train_ds, val_ds, test_ds = random_split(
                dataset, [train_size, val_size, test_size]
            )

        # Choose appropriate collate function and batch size
        if tokenize:
            collate_fn = dataset.collate_with_tokenization
            batch_size = config.batch_size
            pin_memory = True
            drop_last = True
        else:
            collate_fn = BrepToPyDataset.collate
            batch_size = config.batch_size
            pin_memory = False
            drop_last = False

        # Create dataloaders
        train_dl = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=config.data.num_workers if tokenize else config.num_workers,
            collate_fn=collate_fn,
            pin_memory=pin_memory,
            drop_last=drop_last,
        )
        val_dl = DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=config.data.num_workers if tokenize else config.num_workers,
            collate_fn=collate_fn,
            pin_memory=pin_memory,
            drop_last=drop_last if tokenize else False,
        )
        test_dl = DataLoader(
            test_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=config.data.num_workers if tokenize else config.num_workers,
            collate_fn=collate_fn,
            pin_memory=pin_memory,
            drop_last=drop_last,
        )
        return train_dl, val_dl, test_dl
