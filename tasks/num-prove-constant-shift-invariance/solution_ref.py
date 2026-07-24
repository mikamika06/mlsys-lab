import numpy as np

def _stable_softmax(x):
    """Numerically stable softmax: subtract max, exponentiate, normalize."""
    x = np.asarray(x, dtype=np.float64)
    max_x = np.max(x, axis=-1, keepdims=True)
    e_x = np.exp(x - max_x)
    return e_x / np.sum(e_x, axis=-1, keepdims=True)

def softmax_shift_invariant(logits, shift):
    """
    Returns the maximum absolute error between softmax(logits) and
    softmax(logits - shift), proving numerical invariance to constant shifts.
    """
    logits = np.asarray(logits, dtype=np.float64)
    shift = np.asarray(shift, dtype=np.float64)

    soft_original = _stable_softmax(logits)
    soft_shifted = _stable_softmax(logits - shift)

    return float(np.max(np.abs(soft_original - soft_shifted)))
