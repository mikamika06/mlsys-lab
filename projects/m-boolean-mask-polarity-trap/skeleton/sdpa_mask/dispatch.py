import torch
import torch.nn.functional as F
from sdpa_mask.masking import create_sdpa_boolean_mask, invert_mask_if_needed


def run_sdpa_with_mask(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    lengths: torch.Tensor,
    input_mask_convention: str = "keep_true"
) -> torch.Tensor:
    """Execute SDPA ensuring correct boolean mask polarity."""
    raise NotImplementedError
