import torch


def measure_ratios(model: torch.nn.Module, example_inputs: list[torch.Tensor]) -> dict[str, float]:
    raise NotImplementedError
