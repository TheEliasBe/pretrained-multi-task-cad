# modeling_brepcadcoder.py
import torch
from transformers import PreTrainedModel

from .cadcoder_brep import BrepCadcoder
from .configuration_brepcadcoder import BrepCadcoderConfig


class BrepCadcoderHF(PreTrainedModel):
    config_class = BrepCadcoderConfig
    base_model_prefix = "brepcadcoder"

    def __init__(self, config: BrepCadcoderConfig, **unused):
        super().__init__(config)
        # Compose your trained module inside the HF wrapper
        self.model = BrepCadcoder(config)

        # ensure parameters are registered so save_pretrained() captures them
        self.post_init()

    # delegate to your BaseModel forward (already implemented in BaseModel)
    def forward(self, *args, **kwargs):
        return self.model.forward(*args, **kwargs)

    # optional: delegate generate if your BaseModel implements it
    @torch.no_grad()
    def generate(self, *args, **kwargs):
        if hasattr(self.model, "generate"):
            return self.model.generate(*args, **kwargs)
        raise NotImplementedError("generate not implemented in BaseModel")
