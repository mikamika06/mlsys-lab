import numpy as np


def packed_attention_with_reset_mask(Q: np.ndarray, K: np.ndarray, V: np.ndarray, segment_ids: np.ndarray) -> np.ndarray:
    """Causal self-attention over multiple documents PACKED into one
    training sequence, with the mask RESET at every segment boundary.

    Q, K, V: (n, d). segment_ids: (n,) int array; segment_ids[i] is the
    segment/document index token i belongs to (e.g. [0,0,0,1,1,2,2,2,2] for
    three packed documents of length 3, 2, 4).

    Row i may attend to column j iff j <= i (causal) AND segment_ids[j] ==
    segment_ids[i] (same document -- the mask resets, exactly like resetting
    position ids at each packed-document boundary). Returns (n, d).
    """
    n, d = Q.shape
    scores = (Q.astype(np.float64) @ K.astype(np.float64).T) / np.sqrt(d)

    row = np.arange(n)[:, None]
    col = np.arange(n)[None, :]
    same_seg = segment_ids[:, None] == segment_ids[None, :]
    causal = col <= row
    allowed = same_seg & causal

    scores = np.where(allowed, scores, -np.inf)
    scores = scores - np.max(scores, axis=1, keepdims=True)
    probs = np.exp(scores)
    probs = probs / np.sum(probs, axis=1, keepdims=True)
    return probs @ V.astype(np.float64)
