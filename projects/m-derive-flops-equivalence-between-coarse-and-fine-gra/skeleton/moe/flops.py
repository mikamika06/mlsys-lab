import numpy as np


def compute_moe_ffn_flops(d_model, d_ffn, num_tokens):
    """Computes total FLOPs for standard dense SwiGLU FFN or expert execution per token."""
    raise NotImplementedError


def derive_fine_grained_split(d_model, d_ffn_coarse, num_coarse, k_coarse, num_shared, k_fine, split_factor):
    """Derives fine-grained expert d_ffn_fine and returns FLOP equivalence breakdown."""
    raise NotImplementedError
