import torch
import torch.nn as nn


class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        raise NotImplementedError


def structural_prune(model, prune_ratio):
    raise NotImplementedError


def measure_speedup(model_orig, model_pruned, sample_input):
    raise NotImplementedError
