import numpy as np


def varlen_block_diagonal_attention(q: np.ndarray, k: np.ndarray, v: np.ndarray,
                                     cu_seqlens: np.ndarray) -> np.ndarray:
    """Packed varlen self-attention: several sequences packed row-wise
    into (N, d) tensors, boundaries given by `cu_seqlens` (the standard
    FlashAttention-varlen / xformers convention). Each token attends only
    within its own segment (full, non-causal attention inside a segment,
    none across segments).

    q, k, v    : (N, d).
    cu_seqlens : (n_seqs + 1,) int, cu_seqlens[0] == 0,
                 cu_seqlens[-1] == N. Sequence i occupies rows
                 cu_seqlens[i] : cu_seqlens[i+1].

    Returns (N, d).
    """
    q = np.asarray(q, dtype=np.float64)
    k = np.asarray(k, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    cu_seqlens = np.asarray(cu_seqlens, dtype=np.int64)

    N, d = q.shape
    boundaries = cu_seqlens[1:]  # end (exclusive) of each segment
    seq_id = np.searchsorted(boundaries, np.arange(N), side="right")

    same_seq = seq_id[:, None] == seq_id[None, :]  # (N, N)

    scores = (q @ k.T) / np.sqrt(d)
    scores = np.where(same_seq, scores, -np.inf)

    scores = scores - np.max(scores, axis=1, keepdims=True)
    probs = np.exp(scores)
    probs = probs / np.sum(probs, axis=1, keepdims=True)
    return probs @ v
