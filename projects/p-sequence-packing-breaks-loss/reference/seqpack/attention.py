import numpy as np


def create_block_diagonal_mask(seq_ids):
    """Creates a causal block-diagonal mask for packed sequences."""
    s_ids = np.asarray(seq_ids, dtype=np.int64)
    L = len(s_ids)
    i_idx, j_idx = np.tril_indices(L)
    causal_mask = np.zeros((L, L), dtype=bool)
    causal_mask[i_idx, j_idx] = True

    same_seq = (s_ids[:, None] == s_ids[None, :])
    valid_seq = (s_ids[:, None] >= 0) & (s_ids[None, :] >= 0)

    return causal_mask & same_seq & valid_seq


def compute_packed_attention(query, key, value, seq_ids):
    """Computes scaled dot-product attention with block-diagonal masking."""
    Q = np.asarray(query, dtype=np.float64)
    K = np.asarray(key, dtype=np.float64)
    V = np.asarray(value, dtype=np.float64)

    L, D = Q.shape
    scale = 1.0 / np.sqrt(D)
    scores = np.matmul(Q, K.T) * scale

    mask = create_block_diagonal_mask(seq_ids)
    masked_scores = np.where(mask, scores, -1e9)

    row_max = np.max(masked_scores, axis=-1, keepdims=True)
    exp_scores = np.exp(masked_scores - row_max)
    exp_scores = np.where(mask, exp_scores, 0.0)

    row_sums = np.sum(exp_scores, axis=-1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1.0, row_sums)

    attn_weights = exp_scores / row_sums
    output = np.matmul(attn_weights, V)

    return output, attn_weights
