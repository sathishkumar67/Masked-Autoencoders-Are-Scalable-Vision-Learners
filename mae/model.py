from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple
import gin
import torch
import torch.nn as nn
import inspect
from typing import Optional, Tuple
from mae.encoder import EncoderConfig, EncoderModel
from mae.decoder import DecoderConfig, DecoderModel

@gin.configurable
@dataclass
class MAEConfig:
    lr: float 
    batch_size: int
    epochs: int
    weight_decay: float
    eps: float
    betas: Tuple[float, float]
    gpu_count: int
    num_workers: int
    pin_memory: bool
    clip_grad_norm_val: float
    training_backend: str
    eta_min: float
    gradient_accumulation_steps: int
    local_dir: str = "/kaggle/working"
    encoder_config_path: str = f"/kaggle/working/config/encoder_config1.gin"
    decoder_config_path: str = "/kaggle/working/config/decoder_config1.gin"
    dtype: torch.dtype = torch.bfloat16
    fused_optimizer: bool = "fused" in inspect.signature(torch.optim.AdamW).parameters
    model_device: Optional[str|torch.device] = torch.device("cpu")    
    
    warmup_steps: Optional[int] = None
    warmup_steps_ratio: Optional[float] = 0.15 
    total_steps: Optional[int] = None
    steps_per_epoch: Optional[int] = None  


    def __post_init__(self) -> None:
        pass
    
    
class MAE(nn.Module):
    def __init__(self, config: MAEConfig) -> None:
        super().__init__()
        self.config = config
        # Load the encoder and decoder configs
        gin.parse_config_file(f"{config.encoder_config_path}")
        encoder_config = EncoderConfig()
        gin.parse_config_file(f"{config.decoder_config_path}")
        decoder_config = DecoderConfig()

        self.encoder = EncoderModel(encoder_config)
        self.decoder = DecoderModel(decoder_config)

    def forward(self, x) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        encoder_op, mask, ids_restore = self.encoder(x)
        x_rec, loss, ids_restore, mask = self.decoder((encoder_op, mask, ids_restore), x)
        return x_rec, loss, ids_restore, mask