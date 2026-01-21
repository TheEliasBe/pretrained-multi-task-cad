# configuration_brepcadcoder.py
from transformers import PretrainedConfig


class BrepCadcoderConfig(PretrainedConfig):
    model_type = "brepcadcoder"

    def __init__(
        self,
        decoder_model_name="org/decoder-base",
        encoder_hidden_size=768,
        pad_token_id=0,
        eos_token_id=2,
        **kwargs,
    ):
        super().__init__(pad_token_id=pad_token_id, eos_token_id=eos_token_id, **kwargs)
        self.decoder_model_name = decoder_model_name
        self.encoder_hidden_size = encoder_hidden_size
