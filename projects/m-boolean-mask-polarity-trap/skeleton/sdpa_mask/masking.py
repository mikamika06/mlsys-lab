import torch


def create_sdpa_boolean_mask(lengths: torch.Tensor, max_len: int) -> torch.Tensor:
    """Create boolean attn_mask compatible with PyTorch SDPA."""
    raise NotImplementedError


def invert_mask_if_needed(mask: torch.Tensor, target_convention: str) -> torch.Tensor:
    """Invert mask polarity if target convention differs."""
    raise NotImplementedError
