"""
DeepSeek Coder model for prompt + partial code → complete code generation.

This model fine-tunes DeepSeek Coder (instruct variant) for code completion
given a natural language prompt and partial Python code.
"""

import pathlib
from typing import Optional

import torch
import torch.nn as nn

import wandb
from models.cadcoder_base import BaseModel
from models.config import Config


class DeepSeekCodeCompletion(BaseModel):
    """
    DeepSeek Coder model for code completion from prompts and partial code.

    Architecture:
    - Uses DeepSeek Coder Instruct as base model
    - Optional LoRA for efficient fine-tuning
    - Trains with causal language modeling objective
    - Only computes loss on the completion part (not prompt/partial code)
    """

    def __init__(self, config: Config):
        """
        Initialize the model.

        Args:
            config: Configuration object with model parameters
        """
        super().__init__(config)
        self.config = config
        # Logging tables for W&B
        self.training_samples = wandb.Table(
            columns=["step", "prompt", "partial_code", "generated", "target", "loss"]
        )
        self.validation_samples = wandb.Table(
            columns=["prompt", "partial_code", "generated", "target"]
        )
        self.prefix_encoder = nn.Module()

    def get_prefix_embeds(self, batch, batch_idx=None):
        return torch.empty((self.config.batch_size,), device=self.device)

    def generate(
        self,
        prompt: str,
        partial_code: str = "",
        max_new_tokens: Optional[int] = None,
        temperature: float = 0.2,
        top_k: int = 10,
        top_p: float = 0.9,
    ) -> str:
        """
        Generate code completion given prompt and partial code.

        Args:
            prompt: Natural language description
            partial_code: Partial Python code (optional)
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            top_p: Top-p (nucleus) sampling parameter

        Returns:
            Generated code completion
        """
        self.model.eval()

        # Format input
        if partial_code:
            input_text = f"""### Instruction:
{prompt}

### Code:
{partial_code}

### Response:
"""
        else:
            input_text = f"""### Instruction:
{prompt}

### Response:
"""

        # Tokenize
        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            add_special_tokens=True,
        ).to(self.device)

        # Generate
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens or self.config.inference.max_new_tokens,
                do_sample=temperature > 0,
                temperature=temperature if temperature > 0 else 1.0,
                top_k=top_k,
                top_p=top_p,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        # Decode only the generated part
        generated = self.tokenizer.decode(
            output_ids[0, inputs["input_ids"].shape[1] :],
            skip_special_tokens=True,
        )

        return generated

    def save_pretrained(self, save_path: str):
        """
        Save the model and tokenizer.

        Args:
            save_path: Directory to save the model
        """
        save_path = pathlib.Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)

        # Save model
        if hasattr(self.model, "save_pretrained"):
            self.model.save_pretrained(save_path)
        else:
            # For PEFT models
            self.model.save_pretrained(save_path)

        # Save tokenizer
        self.tokenizer.save_pretrained(save_path)

        print(f"Model saved to {save_path}")
