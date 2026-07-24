import numpy as np


def paged_fp8_attention(k_codes_phys, v_codes_phys, k_scale, v_scale, block_table, seq_len, q):
    """Gather paged FP8 (E4M3FN) KV via block_table, truncate to seq_len,
    dequantize per head, and run scaled dot-product attention.

    k_codes_phys, v_codes_phys: uint8 arrays (num_phys_blocks, block_size, n_heads, D)
    k_scale, v_scale: float arrays (n_heads,)
    block_table: int array (L_b,) physical block index per logical block
    seq_len: int, true number of valid logical tokens
    q: float array (n_heads, D)

    Returns float64 array (n_heads, D).
    """
    raise NotImplementedError('your code here')
