import torch
import torch.nn as nn
from transformers import AutoImageProcessor, AutoModel


class ImageEmbedder(nn.Module):
    """
    Encodes an image into `num_virtual_tokens` prefix tokens in `dim_model` space.
    Defaults to ViT; swap `model_name` for CLIP/DINO/etc. if desired.
    """

    def __init__(self, config, model_name: str = "google/vit-base-patch16-224-in21k"):
        super().__init__()
        self.config = config
        self.model_name = getattr(config.cadcoder, "image_model_name", model_name)

        # HF vision backbone
        self.processor = AutoImageProcessor.from_pretrained(self.model_name)
        self.backbone = AutoModel.from_pretrained(self.model_name)  # e.g., ViTModel
        hidden_size = self.backbone.config.hidden_size

        # Projection to prefix-token space
        self.adaptive_layer = nn.Sequential(
            nn.Linear(
                hidden_size * self.config.cadcoder.num_virtual_tokens,
                self.config.decoder.dim_hidden,
            ),
            nn.ReLU(),
            nn.Dropout(self.config.decoder.dropout),
            nn.Linear(
                self.config.decoder.dim_hidden,
                self.config.cadcoder.num_virtual_tokens * self.config.decoder.dim_model,
            ),
            nn.Tanh(),
        )

        # Optionally freeze the image encoder
        self.freeze_encoder = getattr(
            self.config.cadcoder, "freeze_image_encoder", True
        )
        if self.freeze_encoder:
            for p in self.backbone.parameters():
                p.requires_grad = False
            self.backbone.eval()

    def _select_tokens(self, seq: torch.Tensor) -> torch.Tensor:
        """
        Ensure exactly `num_virtual_tokens` tokens per sample by truncation or zero-pad.
        seq: [B, S, D]
        return: [B, num_virtual_tokens, D]
        """
        B, S, D = seq.shape
        N = self.config.cadcoder.num_virtual_tokens
        if S >= N:
            return seq[:, :N, :]
        pad = torch.zeros(B, N - S, D, device=seq.device, dtype=seq.dtype)
        return torch.cat([seq, pad], dim=1)

    def forward(self, images):
        """
        images: list of PIL images, or a float tensor [B, C, H, W] in range [0,1] or [0,255].
        returns: prefix tokens [B, N, dim_model]
        """
        device = next(self.parameters()).device
        # Let the processor handle PIL or tensors
        proc = self.processor(images=images, return_tensors="pt")
        pixel_values = proc["pixel_values"].to(device)

        with torch.no_grad() if self.freeze_encoder else torch.enable_grad():
            out = self.backbone(pixel_values=pixel_values)
            # ViT-like models: last_hidden_state = [B, 1 + num_patches, hidden]
            # Drop CLS ([ :, 0 ]) to use patch tokens for prefix construction.
            tokens = out.last_hidden_state
            if tokens.size(1) > 1:
                tokens = tokens[:, 1:, :]  # remove CLS

        selected = self._select_tokens(tokens)  # [B, N, hidden]
        flat = selected.reshape(selected.size(0), -1)  # [B, N*hidden]
        projected = self.adaptive_layer(flat)  # [B, N*dim_model]
        prefix = projected.view(
            projected.size(0),
            self.config.cadcoder.num_virtual_tokens,
            self.config.decoder.dim_model,
        )  # [B, N, dim_model]
        return prefix
