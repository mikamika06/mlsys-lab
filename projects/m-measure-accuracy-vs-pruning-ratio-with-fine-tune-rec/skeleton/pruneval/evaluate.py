import torch


def evaluate_accuracy(model: torch.nn.Module, dataloader) -> float:
    raise NotImplementedError


def measure_curve(model: torch.nn.Module, dataloader, example_inputs: torch.Tensor, ratios: list) -> dict:
    raise NotImplementedError
