import numpy as np

def attention_roundtrip(Q: np.ndarray,
                        K: np.ndarray,
                        V: np.ndarray):
    """
    Correct implementation of scaled dot‑product attention.
    Returns the raw scores S, the softmax probabilities P,
    and the weighted output O.
    """
    d = Q.shape[1]
    # Raw scaled scores
    S = (Q @ K.T) / np.sqrt(d)
    # Row‑wise softmax with numerical stability
    max_S = np.max(S, axis=-1, keepdims=True)
    exp_S = np.exp(S - max_S)
    P = exp_S / np.sum(exp_S, axis=-1, keepdims=True)
    # Weighted sum of values
    O = P @ V
    return S, P, O
