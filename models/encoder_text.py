import torch
import torch.nn as nn
from transformers import AutoModel


class TextEmbedder(nn.Module):
    def __init__(self, config):
        super(TextEmbedder, self).__init__()
        self.config = config
        self.model = AutoModel.from_pretrained("bert-base-uncased")
        self.adaptive_layer = nn.Sequential(
            nn.Linear(
                768 * self.config.cadcoder.num_virtual_tokens,
                self.config.decoder.dim_hidden,
            ),
            nn.ReLU(),
            nn.Dropout(self.config.decoder.dropout),
            nn.Linear(
                config.decoder.dim_hidden,
                config.cadcoder.num_virtual_tokens * self.config.decoder.dim_model,
            ),
            nn.Tanh(),
        )

    def forward(self, text):
        with torch.no_grad():
            hidden_states = self.model(text).last_hidden_state  # [B, T, 768]

        # Truncate or pad to exactly `num_tokens` per example
        selected = hidden_states[:, : self.config.cadcoder.num_virtual_tokens, :]
        # flatten
        selected = selected.reshape(selected.shape[0], -1)
        projected = self.adaptive_layer(selected)
        prefix = projected.view(
            self.config.training.batch_size,
            self.config.cadcoder.num_virtual_tokens,
            self.config.decoder.dim_model,
        )
        return prefix
