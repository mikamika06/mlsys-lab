import numpy as np

def standard_attention(q, k_contig, v_contig, context_lens):
    """
    Computes standard attention for Q (length 1, decoding phase) against contiguous KV cache.
    q: (batch_size, num_heads, head_dim)
    Returns: out of shape (batch_size, num_heads, head_dim)
    """
    raise NotImplementedError


def paged_attention(q, k_cache, v_cache, block_tables, context_lens):
    """
    Computes attention directly from the paged KV cache without reconstructing contiguous tensors.
    Returns: out of shape (batch_size, num_heads, head_dim)
    """
    raise NotImplementedError
