import numpy as np

def causal_mask(logits):
    """
    Return logits with -inf added to all strictly upper‑triangular entries.
    The output is a float64 array of the same shape as ``logits``.
    """
    out = np.asarray(logits, dtype=np.float64).copy()
    upper = np.triu_indices_from(out, k=1)
    out[upper] = -np.inf
    return out
