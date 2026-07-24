import numpy as np


def paged_gqa_attention(k_phys, v_phys, block_table, q, n_kv_heads):
    """Gather logical KV from a paged block table and run GQA attention.

    k_phys, v_phys: (num_phys_blocks, block_size, n_kv_heads, D)
    block_table:    (L_b,) physical block index per logical position
    q:              (n_q_heads, D)
    """
    raise NotImplementedError('your code here')
