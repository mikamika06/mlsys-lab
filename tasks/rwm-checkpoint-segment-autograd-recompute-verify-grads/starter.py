import torch


def checkpoint_segment(
    x: torch.Tensor,
    w1: torch.Tensor,
    b1: torch.Tensor,
    w2: torch.Tensor,
    b2: torch.Tensor,
) -> tuple[torch.Tensor, list[torch.Tensor], int]:
    """Checkpoint a two-layer block and return loss, gradients, and saved count."""
    raise NotImplementedError("your code here")
