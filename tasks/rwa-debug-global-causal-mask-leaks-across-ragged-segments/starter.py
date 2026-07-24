import numpy as np


def ragged_causal_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray, cu_seqlens: np.ndarray) -> np.ndarray:
    """Causal self-attention over a PACKED (ragged) batch.

    Q, K, V: (n, d) -- multiple variable-length sequences concatenated along
    the token axis. cu_seqlens: 1-D int array of length (num_segments + 1)
    giving cumulative sequence boundaries.

    Row i may only attend to keys/values at position j such that j <= i AND
    j is in the same segment as i.

    BUG: this implementation applies a single GLOBAL causal mask (col <= row)
    over the whole packed batch and never looks at `cu_seqlens`. The first
    few tokens of segment i therefore still "see" the tail of segment i-1
    (and any earlier segments) -- attention leaks across sequence boundaries.
    """
    n, d = Q.shape
    scores = (Q.astype(np.float64) @ K.astype(np.float64).T) / np.sqrt(d)

    row = np.arange(n)[:, None]
    col = np.arange(n)[None, :]
    scores = np.where(col > row, -np.inf, scores)  # BUG: no per-segment boundary

    scores = scores - np.max(scores, axis=1, keepdims=True)
    probs = np.exp(scores)
    probs = probs / np.sum(probs, axis=1, keepdims=True)
    return probs @ V.astype(np.float64)
