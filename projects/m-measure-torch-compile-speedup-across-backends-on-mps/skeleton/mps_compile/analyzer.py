import torch


def capture_graph_break(model: torch.nn.Module, x: torch.Tensor) -> str:
    raise NotImplementedError


def verify_equivalence(model: torch.nn.Module, x: torch.Tensor) -> bool:
    raise NotImplementedError
