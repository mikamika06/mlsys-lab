import numpy as np


def ragged_causal_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray, cu_seqlens: np.ndarray) -> np.ndarray:
    """Causal self-attention over a PACKED (ragged) batch.

    Q, K, V: (n, d) -- multiple variable-length sequences concatenated along
    the token axis. cu_seqlens: 1-D int array of length (num_segments + 1)
    giving cumulative sequence boundaries, e.g. [0, 3, 7, 10] means segment 0
    is tokens[0:3], segment 1 is tokens[3:7], segment 2 is tokens[7:10].

    Row i may only attend to keys/values at position j such that:
      1. j <= i (causal), AND
      2. j is in the SAME segment as i (no cross-sequence leakage).

    Returns (n, d).
    """
    n, d = Q.shape
    scores = (Q.astype(np.float64) @ K.astype(np.float64).T) / np.sqrt(d)

    row = np.arange(n)[:, None]
    col = np.arange(n)[None, :]

    seg_id = np.zeros(n, dtype=np.int64)
    for s in range(len(cu_seqlens) - 1):
        seg_id[cu_seqlens[s]:cu_seqlens[s + 1]] = s
    same_segment = seg_id[:, None] == seg_id[None, :]
    causal = col <= row
    allowed = same_segment & causal

    scores = np.where(allowed, scores, -np.inf)
    scores = scores - np.max(scores, axis=1, keepdims=True)
    probs = np.exp(scores)
    probs = probs / np.sum(probs, axis=1, keepdims=True)
    return probs @ V.astype(np.float64)
