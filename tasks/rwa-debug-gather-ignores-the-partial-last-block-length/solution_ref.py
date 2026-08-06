import numpy as np


def gathered_attention(k_phys, v_phys, block_table, seq_len, q):
    """Gather logical KV from a block table, truncate to the true seq_len,
    and compute single-query scaled dot-product attention."""
    H = k_phys.shape[-1]
    k_logical = k_phys[block_table].reshape(-1, H).astype(np.float64)[:seq_len]
    v_logical = v_phys[block_table].reshape(-1, H).astype(np.float64)[:seq_len]
    qf = np.asarray(q, dtype=np.float64)

    scores = (k_logical @ qf) / np.sqrt(H)
    scores = scores - np.max(scores)
    weights = np.exp(scores)
    weights = weights / np.sum(weights)

    return (weights[:, None] * v_logical).sum(axis=0)
