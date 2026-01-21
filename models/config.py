from dataclasses import dataclass
from typing import Literal, Optional

import torch


@dataclass
class GeneralConfig:
    """General architecture configuration"""

    model: str = "GraphCad"
    mode: Literal["train", "inference", "evaluate"] = "train"
    precision: torch.dtype = torch.float32
    fast_dev_run: bool = False


@dataclass
class EncoderConfig:
    """Encoder architecture configuration"""

    n_encoder_layer: int = 4
    freeze_encoder: bool = False
    freeze_encoder_epoch: int = 20


@dataclass
class DecoderConfig:
    """Decoder architecture configuration"""

    embedding: str = "PositionalEncodingSinusoidalTokenTypeMask"
    decoder_layer: str = "TransformerDecoderLayerCrossAttention"
    activation_function: str = "swiglu"
    use_memory_pretraining: bool = True
    args_dim: int = 256
    bias: bool = True
    block_size: int = 512
    condense_ratio: int = 1
    dim_hidden: int = 2048  # dimension of FFN layers
    dim_memory: int = 512  # latent dim / dimension of the memory vector
    dim_model: int = 768
    dim_latent: int = 256
    dropout: float = 0.1
    embedding_dim: int = 512
    intermediate_size: Optional[int] = None
    n_head: int = 8
    n_layer: int = 4
    ff_mult: int = 4
    n_query_groups: Optional[int] = None
    norm_eps: float = 1e-5
    num_encoder_layers: int = 4
    parallel_residual: bool = True
    rotary_percentage: float = 0.25
    shared_attention_norm: bool = False
    use_vq: bool = True
    vq_decay: float = 0.99
    use_prefix_mask: int = 0


@dataclass
class OptimizerConfig:
    """Optimizer configuration"""

    optimizer: str = "Adam"
    learning_rate: float = 1e-6
    weigth_decay: float = 1e-4
    scheduler: str = "CosineWithWarmup"
    step_size: int = 10
    gamma: int = 0.1
    gradient_clip_val: float = 1.0
    accumulate_grad_batches: int = 2
    gradient_checkpointing: bool = False
    warmup_steps: int = 1000
    betas: tuple = (0.9, 0.98)


@dataclass
class TrainingConfig:
    """Training configuration"""

    max_epochs: int = 2
    batch_size: int = 1
    device: str = "cuda"  # "cuda" if torch.cuda.is_available() else "mps"
    checkpoint_freq: int = 5
    checkpoint: str = None  # "SingleProjectionDecoder-v15.ckpt"
    checkpointing: bool = True
    profiler: Optional[Literal["simple", "advanced", "pytorch"]] = None
    patience: int = 5
    min_delta: float = 0.001
    use_system_prompt: bool = True
    min_lr: float = 1e-6


@dataclass
class DeepspeedConfig:
    """Deepspeed/multi-GPU configuration"""

    multi_gpu: bool = False
    zero_stage: int = 2
    num_gpus: int = 4
    num_nodes: int = 1


@dataclass
class LossConfig:
    """Loss configuration"""

    weight_decay: float = 1e-4
    label_smoothing: float = 0.01
    use_alignment_loss: bool = True
    alignment_loss_weight: float = 0.1


@dataclass
class DataConfig:
    """Data configuration"""

    num_workers: int = 0
    dataset_size: int = 2000
    max_total_len: int = 4096
    min_total_len: int = 0
    seed: int = 42
    shuffle: bool = False
    drop_last: bool = False
    unused_param: bool = False
    use_cad_query_code: bool = True
    root_dir: str = "./data"  # Update to your local data directory
    partial_ratio: float = 0.5
    train_ratio: float = 0.8
    val_ratio: float = 0.1


@dataclass
class InferenceConfig:
    """Inference configuration"""

    beam_width: int = 5
    temperature: float = 0.1
    top_p: int = 0.90
    top_k: int = 1
    context_len: int = 30
    alpha: int = 0.6
    max_new_tokens: int = 1024
    use_beam_pruning: bool = False


@dataclass
class LoggingConfig:
    """Logging and visualization configuration"""

    dpi: int = 150
    log_every_n_steps: int = 10
    log_one_sample: bool = False
    log_attention_scores: bool = False
    log_alignment: bool = False
    logger: str = "mlop"
    logger_mode: str = "disabled"
    logger_tag: Optional[str] = None


@dataclass
class BrepEncoderConfig:
    """BREP encoder configuration"""

    dim_graph_embedding: int = 768  # Will be set to decoder.dim_model
    crv_in_channels: int = 3
    crv_emb_dim: int = 256
    srf_emb_dim: int = 256
    curv_u_samples: int = 10
    surf_u_samples: int = 10
    surf_v_samples: int = 10
    node_features: int = 256
    edge_features: int = 256
    num_embeddings: int = 512
    commitment_cost: float = 0.25
    n_vq_embeddings: int = 512
    n_encoder_layer: int = 4
    max_nodes: int = 200
    max_edges: int = 800


@dataclass
class Text2CadConfig:
    """Text2CAD specific configuration"""

    max_seq_len_text: int = 512
    tokenizer_model_name: str = "bert-large-uncased"
    pretrained_decoder_path: str = "SingleProjectionDecoder-v34.ckpt"
    noise: float = 0.1
    max_tokens: int = 512


@dataclass
class CadcoderConfig:
    """CADCoder specific configuration"""

    num_virtual_tokens: int = 20
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: Optional[list] = None
    use_lora: bool = True
    model_name: str = "bigcode/starcoder2-7b"
    quantization: bool = True
    use_bos_token: bool = True
    attn_implementation: str = "sdpa"
    use_pretrained: bool = True


@dataclass
class MemoryConfig:
    """Memory optimization configuration"""

    empty_cache_freq: int = 0


@dataclass
class Config:
    """Main configuration class containing all sub-configurations"""

    general: GeneralConfig = None
    encoder: EncoderConfig = None
    decoder: DecoderConfig = None
    optimizer: OptimizerConfig = None
    training: TrainingConfig = None
    deepspeed: DeepspeedConfig = None
    loss: LossConfig = None
    data: DataConfig = None
    inference: InferenceConfig = None
    logging: LoggingConfig = None
    brep_encoder: BrepEncoderConfig = None
    text2cad: Text2CadConfig = None
    cadcoder: CadcoderConfig = None
    memory: MemoryConfig = None

    def __post_init__(self):
        """Initialize sub-configs with defaults if not provided"""
        if self.general is None:
            self.general = GeneralConfig()
        if self.encoder is None:
            self.encoder = EncoderConfig()
        if self.decoder is None:
            self.decoder = DecoderConfig()
        if self.optimizer is None:
            self.optimizer = OptimizerConfig()
        if self.training is None:
            self.training = TrainingConfig()
        if self.deepspeed is None:
            self.deepspeed = DeepspeedConfig()
        if self.loss is None:
            self.loss = LossConfig()
        if self.data is None:
            self.data = DataConfig()
        if self.inference is None:
            self.inference = InferenceConfig()
        if self.logging is None:
            self.logging = LoggingConfig()
        if self.brep_encoder is None:
            self.brep_encoder = BrepEncoderConfig()
        if self.text2cad is None:
            self.text2cad = Text2CadConfig()
        if self.cadcoder is None:
            self.cadcoder = CadcoderConfig()
        if self.memory is None:
            self.memory = MemoryConfig()

        # Set dim_graph_embedding to match decoder.dim_model
        self.brep_encoder.dim_graph_embedding = self.decoder.dim_model

    # Backward compatibility properties for existing code
    @property
    def model(self):
        return self.general.model

    @property
    def mode(self):
        return self.general.mode

    @property
    def precision(self):
        return self.general.precision

    @property
    def fast_dev_run(self):
        return self.general.fast_dev_run

    @property
    def max_epochs(self):
        return self.training.max_epochs

    @property
    def batch_size(self):
        return self.training.batch_size

    @property
    def device(self):
        return self.training.device

    @property
    def checkpoint(self):
        return self.training.checkpoint

    @property
    def checkpointing(self):
        return self.training.checkpointing

    @property
    def learning_rate(self):
        return self.optimizer.learning_rate

    @property
    def gradient_clip_val(self):
        return self.optimizer.gradient_clip_val

    @property
    def accumulate_grad_batches(self):
        return self.optimizer.accumulate_grad_batches

    @property
    def multi_gpu(self):
        return self.deepspeed.multi_gpu

    @property
    def num_gpus(self):
        return self.deepspeed.num_gpus

    @property
    def num_nodes(self):
        return self.deepspeed.num_nodes

    @property
    def log_every_n_steps(self):
        return self.logging.log_every_n_steps

    @property
    def logger(self):
        return self.logging.logger

    @property
    def logger_mode(self):
        return self.logging.logger_mode

    @property
    def logger_tag(self):
        return self.logging.logger_tag

    @property
    def patience(self):
        return self.training.patience

    @property
    def min_delta(self):
        return self.training.min_delta

    @property
    def max_new_tokens(self):
        return self.inference.max_new_tokens
