import numpy as np


def paged_gqa_attention(k_phys, v_phys, block_table, q, n_kv_heads):
    """Gather logical KV from a paged block table and run GQA attention.

    k_phys, v_phys: (num_phys_blocks, block_size, n_kv_heads, D)
    block_table:    (L_b,) physical block index per logical position
    q:              (n_q_heads, D)
    """
    k_phys = np.asarray(k_phys, dtype=np.float64)
    v_phys = np.asarray(v_phys, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)

    num_phys, block_size, H_kv, D = k_phys.shape
    L_b = block_table.shape[0]

    k_logical = k_phys[block_table].reshape(L_b * block_size, H_kv, D)
    v_logical = v_phys[block_table].reshape(L_b * block_size, H_kv, D)

    n_q_heads = q.shape[0]
    group = n_q_heads // n_kv_heads
    scale = 1.0 / np.sqrt(D)

    out = np.zeros((n_q_heads, D), dtype=np.float64)
    for h in range(n_q_heads):
        kv_h = h // group
        K = k_logical[:, kv_h, :]
        V = v_logical[:, kv_h, :]
        scores = (K @ q[h]) * scale
        scores = scores - scores.max()
        w = np.exp(scores)
        w = w / w.sum()
        out[h] = w @ V

    return out
