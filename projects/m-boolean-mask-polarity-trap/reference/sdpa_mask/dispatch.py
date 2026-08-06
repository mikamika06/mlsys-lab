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
    max_len = key.shape[-2]
    mask = create_sdpa_boolean_mask(lengths, max_len)
    if input_mask_convention == "keep_true":
        sdpa_mask = mask
    elif input_mask_convention == "sdpa":
        sdpa_mask = mask
    else:
        sdpa_mask = invert_mask_if_needed(mask, input_mask_convention)
    return F.scaled_dot_product_attention(query, key, value, attn_mask=sdpa_mask)
