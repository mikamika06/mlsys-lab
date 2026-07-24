import numpy as np


def gather_attention(k_phys, v_phys, block_table, q):
    k_logical = k_phys[block_table].reshape(-1, k_phys.shape[-1]).astype(np.float64)
    v_logical = v_phys[block_table].reshape(-1, v_phys.shape[-1]).astype(np.float64)
    q = q.astype(np.float64)

    scores = (k_logical @ q) / np.sqrt(k_logical.shape[-1])
    scores = scores - np.max(scores)
    weights = np.exp(scores)
    weights = weights / np.sum(weights)

    return np.sum(weights[:, None] * v_logical, axis=0).astype(np.float64)
