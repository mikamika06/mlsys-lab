import torch
import torch.nn as nn


class TinyAttentionModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        raise NotImplementedError

    def forward(self, x, use_sdpa=False):
        raise NotImplementedError
