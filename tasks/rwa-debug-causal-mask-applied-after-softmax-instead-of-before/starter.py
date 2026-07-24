import numpy as np


def causal_self_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray) -> np.ndarray:
    """Causal scaled dot-product self-attention.

    Q, K, V: (n, d). Row i may only attend to keys/values at position
    <= i. Returns (n, d).

    BUG: the causal mask is applied to the softmax PROBABILITIES (zeroing
    disallowed entries after the fact) instead of to the logits before
    softmax, so masked rows no longer sum to 1 -- the output is silently
    under-normalized.
    """
    n, d = Q.shape
    scores = (Q @ K.T) / np.sqrt(d)

    scores = scores - np.max(scores, axis=1, keepdims=True)
    probs = np.exp(scores)
    probs = probs / np.sum(probs, axis=1, keepdims=True)

    row = np.arange(n)[:, None]
    col = np.arange(n)[None, :]
    mask = (col <= row).astype(np.float64)
    probs = probs * mask  # BUG: should have masked the logits pre-softmax

    return probs @ V
