import torch


def triton_silu_forward(x: torch.Tensor, BLOCK_SIZE: int = 1024) -> torch.Tensor:
    raise NotImplementedError


def triton_silu_backward(x: torch.Tensor, grad_output: torch.Tensor, BLOCK_SIZE: int = 1024) -> torch.Tensor:
    raise NotImplementedError
