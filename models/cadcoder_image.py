from models import BaseModel
from models.encoder_image import ImageEmbedder


class CadcoderImage(BaseModel):
    """
    Image-conditioned Cadcoder that supplies prefix embeddings from an image encoder.
    Expects batch["image"] (PIL list or tensor) in `get_prefix_embeds`.
    """

    def __init__(self, config):
        super().__init__(config)
        self.config = config

        self.prefix_encoder = ImageEmbedder(config)

    def get_prefix_embeds(self, batch):
        images = batch["image"]  # list of PIL Images or a tensor [B, C, H, W]
        prefix_embeds = self.prefix_encoder(images)
        return prefix_embeds
