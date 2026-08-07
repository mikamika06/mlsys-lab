import torch


def compute_pruned_model(model: torch.nn.Module, example_inputs: torch.Tensor, ratio: float) -> torch.nn.Module:
    raise NotImplementedError
