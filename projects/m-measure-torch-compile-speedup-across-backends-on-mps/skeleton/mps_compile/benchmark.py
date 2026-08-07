import torch


def measure_speedup(model: torch.nn.Module, x: torch.Tensor) -> float:
    raise NotImplementedError
