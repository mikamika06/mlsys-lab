import numpy as np


def create_block_diagonal_mask(seq_ids):
    """Creates a causal block-diagonal mask for packed sequences."""
    raise NotImplementedError


def compute_packed_attention(query, key, value, seq_ids):
    """Computes scaled dot-product attention with block-diagonal masking."""
    raise NotImplementedError
