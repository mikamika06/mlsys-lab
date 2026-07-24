import numpy as np


def packed_attention_with_reset_mask(Q: np.ndarray, K: np.ndarray, V: np.ndarray, segment_ids: np.ndarray) -> np.ndarray:
    """Causal self-attention over multiple documents PACKED into one
    training sequence, with the mask meant to RESET at every segment
    boundary. Q, K, V: (n, d). segment_ids: (n,) int array giving each
    token's segment/document index.

    BUG: this implementation applies a single GLOBAL causal mask
    (col <= row) over the whole packed sequence and never looks at
    `segment_ids`. Every document therefore still attends into the TAIL of
    the previous document (and any earlier ones) as if it were its own
    left context.
    """
    n, d = Q.shape
    scores = (Q.astype(np.float64) @ K.astype(np.float64).T) / np.sqrt(d)

    row = np.arange(n)[:, None]
    col = np.arange(n)[None, :]
    scores = np.where(col <= row, scores, -np.inf)  # BUG: ignores segment_ids

    scores = scores - np.max(scores, axis=1, keepdims=True)
    probs = np.exp(scores)
    probs = probs / np.sum(probs, axis=1, keepdims=True)
    return probs @ V.astype(np.float64)
