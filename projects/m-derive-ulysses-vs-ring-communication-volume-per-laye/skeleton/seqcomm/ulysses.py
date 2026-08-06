import torch


def ulysses_all_to_all(
    tensor: torch.Tensor, scatter_dim: int, gather_dim: int, group=None
) -> torch.Tensor:
    """Perform All-to-All tensor redistribution across sequence parallel rank group."""
    raise NotImplementedError


def ulysses_attention_forward(
    query_chunk: torch.Tensor,
    key_chunk: torch.Tensor,
    value_chunk: torch.Tensor,
    num_heads: int,
    group=None,
) -> torch.Tensor:
    """Compute sequence parallel attention using DeepSpeed-Ulysses pattern."""
    raise NotImplementedError
