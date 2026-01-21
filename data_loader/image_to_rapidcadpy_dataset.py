import io
import pathlib
import re
import tokenize
from io import BytesIO
from typing import List

import pandas as pd
import torch
from line_profiler import profile
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split
from transformers import AutoTokenizer


class ImageToPyDataset(Dataset):
    def __init__(self, config):
        """
        Args:
            config: configuration object
            records: list of dicts with {"id": str, "bytes": bytes, "code": str}
            tokenize: whether to tokenize code on init
        """
        self.config = config

        # tokenizer setup
        self.code_tokenizer = AutoTokenizer.from_pretrained(
            config.cadcoder.model_name, trust_remote_code=True
        )
        self.code_tokenizer.pad_token = self.code_tokenizer.eos_token

        # Load all parquet files in data_dir and concat, limit to dataset_size
        data_dir = pathlib.Path(config.data.root_dir) / "GenCAD-Code" / "data"
        parquet_files = sorted(data_dir.glob("*.parquet"))
        dfs = []
        total_rows = 0
        for pf in parquet_files:
            df = pd.read_parquet(pf)
            if (
                self.config.data.dataset_size
                and total_rows + len(df) > self.config.data.dataset_size
            ):
                df = df.iloc[: self.config.data.dataset_size - total_rows]
            dfs.append(df)
            total_rows += len(df)
            if (
                self.config.data.dataset_size
                and total_rows >= self.config.data.dataset_size
            ):
                break
        cadquery_df = pd.concat(dfs, ignore_index=True)

        cadquery_df["cadquery"] = cadquery_df["cadquery"].apply(
            lambda c: self.round_floats_in_code_tokenized(c, digits=2)
        )
        tokenized_df = cadquery_df["cadquery"].apply(self.tokenize_code)

        # robust expansion to columns
        tokenized_df = tokenized_df.apply(
            pd.Series
        )  # -> DataFrame with keys as columns

        cadquery_df[["input_ids", "attention_mask", "labels"]] = tokenized_df[
            ["input_ids", "attention_mask", "labels"]
        ]
        self.data = cadquery_df.to_dict("records")

    def _load_image(self, img_bytes: dict) -> Image.Image:
        # 2. extract the actual PNG bytes
        # if they are in base64, decode first; if already raw binary (escaped in JSON), wrap with bytes()
        raw = img_bytes["bytes"]
        if isinstance(raw, str):
            # JSON escaped string → convert back to bytes
            raw = raw.encode(
                "latin1"
            )  # or use base64.b64decode(raw) if it was base64-encoded

        # 3. load with PIL
        img = Image.open(BytesIO(raw))
        if img.mode != "RGB":
            img = img.convert("RGB")
        return img

    def tokenize_code(self, code: str):
        # tokenize with HF
        tok = self.code_tokenizer(
            code,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.config.data.max_total_len,
            add_special_tokens=True,
            padding_side="right",
        )
        # labels: -100 for prefix tokens
        pad_labels = torch.full(
            (1, self.config.cadcoder.num_virtual_tokens), -100, dtype=torch.long
        )
        labels = torch.cat([pad_labels, tok["input_ids"]], dim=1)
        first_eos = (labels == self.code_tokenizer.eos_token_id).cumsum(dim=1)
        labels[first_eos > 1] = -100

        attention_mask = torch.cat(
            [
                torch.ones(
                    (1, self.config.cadcoder.num_virtual_tokens), dtype=torch.long
                ),
                tok["attention_mask"],
            ],
            dim=1,
        )

        return {
            "input_ids": tok["input_ids"].squeeze(0),
            "attention_mask": attention_mask.squeeze(0),
            "labels": labels.squeeze(0),
        }

    @staticmethod
    def round_floats_in_code(code: str, digits: int = 2) -> str:
        float_pattern = re.compile(r"-?\d+\.\d+")

        def round_match(m):
            num = float(m.group())
            return str(round(num, digits))

        return float_pattern.sub(round_match, code)

    @staticmethod
    def round_floats_in_code_tokenized(code: str, digits: int = 6) -> str:
        # build absolute offsets per line
        lines = code.splitlines(keepends=True)
        line_starts = []
        s = 0
        for ln in lines:
            line_starts.append(s)
            s += len(ln)

        repls = []  # (abs_start, abs_end, new_str)

        for tok in tokenize.generate_tokens(io.StringIO(code).readline):
            if tok.type != tokenize.NUMBER:
                continue
            txt = tok.string
            # keep integers
            if re.fullmatch(r"[-+]?\d+", txt):
                continue
            try:
                v = float(txt)
            except ValueError:
                continue
            out = f"{round(v, digits):.{digits}f}".rstrip("0").rstrip(".")
            if out in ("", "-0", "+0"):
                out = "0"

            # absolute positions
            (srow, scol), (erow, ecol) = tok.start, tok.end
            abs_s = line_starts[srow - 1] + scol
            abs_e = line_starts[erow - 1] + ecol
            repls.append((abs_s, abs_e, out))

        # apply replacements back-to-front to keep indices valid
        out_code = code
        for abs_s, abs_e, new in sorted(repls, key=lambda x: x[0], reverse=True):
            out_code = out_code[:abs_s] + new + out_code[abs_e:]
        return out_code

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        rec = self.data[idx]
        img = self._load_image(rec["image"])
        sample = {
            "image": img,
            "code": rec["cadquery"],
        }
        if "input_ids" in rec:
            sample.update(
                {
                    "input_ids": rec["input_ids"],
                    "attention_mask": rec["attention_mask"],
                    "labels": rec["labels"],
                    "ids": rec["deepcad_id"],
                }
            )
        return sample

    # ---------- splits and collate ----------

    @staticmethod
    def create_splits(config, tokenize: bool = True, **ds_kwargs):
        dataset = ImageToPyDataset(config)
        train_ds, val_ds, test_ds = random_split(dataset, [0.8, 0.1, 0.1])

        collate_fn = dataset.collate_with_tokenization if tokenize else dataset.collate

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

    @profile
    def collate(self, batch):
        """
        On-the-fly tokenization for code. Keeps images as PIL list.
        """
        self.code_tokenizer.pad_token = self.code_tokenizer.eos_token

        images: List[Image.Image] = [b["image"] for b in batch]
        codes = [b["code"] for b in batch]
        ids = [b["id"] for b in batch]

        tok_code = self.code_tokenizer(
            codes,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.config.data.max_total_len,
            add_special_tokens=True,
        )

        # labels with -100 over prefix region
        pad_labels = torch.full(
            (len(batch), self.config.cadcoder.num_virtual_tokens),
            -100,
            dtype=torch.long,
        )
        labels = torch.cat([pad_labels, tok_code["input_ids"]], dim=1)
        first_eos = (labels == self.code_tokenizer.eos_token_id).cumsum(dim=1)
        labels[first_eos > 1] = -100

        return {
            "image": images,  # list of PIL for the model's processor
            "input_ids": tok_code["input_ids"],
            "attention_mask": tok_code["attention_mask"],
            "code": codes,
            "id": ids,
            "labels": labels,
        }

    def collate_with_tokenization(self, batch):
        """
        Uses pre-tokenized code. Keeps images as PIL list.
        """
        images: List[Image.Image] = [b["image"] for b in batch]
        codes = [b["code"] for b in batch]

        input_ids = torch.stack([b["input_ids"] for b in batch])
        attention_mask = torch.stack([b["attention_mask"] for b in batch])
        labels = torch.stack([b["labels"] for b in batch])
        ids = [b["ids"] for b in batch]

        return {
            "image": images,  # consumed by ImageEmbedder processor
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "code": codes,
            "labels": labels,
            "ids": ids,
        }
