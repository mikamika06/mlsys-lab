import numpy as np


def gathered_attention(k_phys, v_phys, block_table, seq_len, q):
    """Gather logical KV from a block table and compute single-query
    scaled dot-product attention."""
    H = k_phys.shape[-1]
    # BUG: reads a full block_size rows from every logical block, including
    # the partially-filled last one, instead of truncating to seq_len — so
    # stale/garbage rows past the true sequence length leak into attention.
    k_logical = k_phys[block_table].reshape(-1, H).astype(np.float64)
    v_logical = v_phys[block_table].reshape(-1, H).astype(np.float64)
    qf = np.asarray(q, dtype=np.float64)

    scores = (k_logical @ qf) / np.sqrt(H)
    scores = scores - np.max(scores)
    weights = np.exp(scores)
    weights = weights / np.sum(weights)

    return (weights[:, None] * v_logical).sum(axis=0)
