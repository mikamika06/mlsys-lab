import numpy as np


def cow_prefix_attention(q_a, k_a, v_a, q_b, k_b, v_b, shared_prefix_len, block_size):
    """Compare COW prefix block-sharing memory to duplicated KV memory, and
    compute each sequence's own causal self-attention output.

    q_a, k_a, v_a: (L_A, d) float64 queries/keys/values for sequence A.
    q_b, k_b, v_b: (L_B, d) float64 queries/keys/values for sequence B.
    shared_prefix_len: number of leading tokens A and B have in common.
    block_size: KV-cache page size in tokens.

    Returns (size_ratio, out_a, out_b):
      size_ratio = duplicated_blocks / unique_blocks, where duplicated =
        ceil(L_A/block_size) + ceil(L_B/block_size) and unique subtracts
        floor(shared_prefix_len/block_size) fully-shared blocks (capped at
        each sequence's own block count).
      out_a, out_b: standard causal scaled dot-product self-attention
        output for each sequence, computed independently over its own
        full q, k, v (sharing must not change these values).
    """
    raise NotImplementedError('your code here')
