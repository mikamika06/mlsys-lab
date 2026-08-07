import numpy as np


def paged_attention(q, k_pool, v_pool, block_tables, seq_lens, block_size, scale=None):
    """Compute attention using paged physical block lookup."""
    raise NotImplementedError
