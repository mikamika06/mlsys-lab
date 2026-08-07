import torch
import torch.nn as nn


class ToyNet(nn.Module):
    def __init__(self):
        super().__init__()
        raise NotImplementedError


def propagate_channels(pruned_channels):
    raise NotImplementedError
