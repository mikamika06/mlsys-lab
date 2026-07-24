import numpy as np


def gumbel_softmax_relaxed(logits: np.ndarray, g: np.ndarray, tau: float) -> np.ndarray:
    """
    Gumbel-softmax relaxation with externally supplied (fixed) Gumbel
    noise g: softmax((logits + g) / tau, axis=-1), computed in a
    numerically stable way (subtract the row max before exponentiating).
    """
    logits = np.asarray(logits, dtype=np.float64)
    g = np.asarray(g, dtype=np.float64)
    z = (logits + g) / tau
    z = z - np.max(z, axis=-1, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=-1, keepdims=True)
