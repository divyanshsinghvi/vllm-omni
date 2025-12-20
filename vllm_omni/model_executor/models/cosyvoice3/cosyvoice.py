from collections.abc import Iterable

import torch
import torch.nn as nn
from vllm.config import VllmConfig
from vllm.logger import init_logger

logger = init_logger(__name__)


class CosyVoiceModel(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()

    def forward(self):
        pass

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        pass
