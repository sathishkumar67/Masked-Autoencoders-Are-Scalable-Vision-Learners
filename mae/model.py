from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple
import gin
import torch
import torch.nn as nn

@gin.configurable
@dataclass
class MAEConfig:
    lr: float 
    batch_size: int
    epochs: int
    device: str
    weight_decay: float
    eps: float
    seed : int
    betas: Tuple[float, float]
    gpu_count: int
    num_workers: int
    pin_memory: bool

class MAE(nn.Module):
    def __init__(self, encoder: nn.Module, decoder: nn.Module) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        encoder_op, mask, ids_restore = self.encoder(x)
        decoder_op = self.decoder((encoder_op, mask, ids_restore), x)
        return decoder_op
