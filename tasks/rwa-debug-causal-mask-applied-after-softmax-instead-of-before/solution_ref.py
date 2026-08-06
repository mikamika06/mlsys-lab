import numpy as np


def causal_self_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray) -> np.ndarray:
    """Causal scaled dot-product self-attention.

    Q, K, V: (n, d). Row i may only attend to keys/values at position
    <= i. Masking is applied to the LOGITS (score[i, j] = -inf for j > i)
    BEFORE softmax, so every row's probabilities still sum to 1 over the
    positions it is allowed to see. Returns (n, d).
    """
    n, d = Q.shape
    scores = (Q @ K.T) / np.sqrt(d)

    row = np.arange(n)[:, None]
    col = np.arange(n)[None, :]
    scores = np.where(col > row, -np.inf, scores)

    scores = scores - np.max(scores, axis=1, keepdims=True)
    probs = np.exp(scores)
    probs = probs / np.sum(probs, axis=1, keepdims=True)
    return probs @ V
