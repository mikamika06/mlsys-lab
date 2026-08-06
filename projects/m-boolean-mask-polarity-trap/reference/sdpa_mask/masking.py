import torch


def create_sdpa_boolean_mask(lengths: torch.Tensor, max_len: int) -> torch.Tensor:
    """Create boolean attn_mask compatible with PyTorch SDPA."""
    seq_range = torch.arange(max_len, device=lengths.device).unsqueeze(0)
    return seq_range < lengths.unsqueeze(1)


def invert_mask_if_needed(mask: torch.Tensor, target_convention: str) -> torch.Tensor:
    """Invert mask polarity if target convention differs."""
    if mask.dtype == torch.bool:
        if target_convention == "sdpa":
            return mask
        elif target_convention == "keep_true":
            return ~mask
        else:
            raise ValueError(f"Unknown convention: {target_convention}")
    return mask
