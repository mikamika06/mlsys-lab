import numpy as np


def block_diagonal_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray, seq_lens: list[int]) -> np.ndarray:
    """Attention over several variable-length sequences PACKED into one
    (N, d) tensor along the row axis (xformers' BlockDiagonalMask). Each
    sequence attends only to its own rows -- full (non-causal) attention
    within a sequence, zero cross-sequence attention.

    Q, K, V   : (N, d), N == sum(seq_lens).
    seq_lens  : list of positive ints, the length of each packed sequence,
                in order.

    Returns (N, d).
    """
    Q = np.asarray(Q, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    N, d = Q.shape

    seq_id = np.empty(N, dtype=np.int64)
    pos = 0
    for i, L in enumerate(seq_lens):
        seq_id[pos:pos + L] = i
        pos += L

    same_seq = seq_id[:, None] == seq_id[None, :]  # (N, N)

    scores = (Q @ K.T) / np.sqrt(d)
    scores = np.where(same_seq, scores, -np.inf)

    scores = scores - np.max(scores, axis=1, keepdims=True)
    probs = np.exp(scores)
    probs = probs / np.sum(probs, axis=1, keepdims=True)
    return probs @ V
