import math
import numpy as np

def streaming_causal_softmax(logits, mask, block_size):
    """
    Compute a row‑wise softmax over the lower triangular part of ``logits``
    (causal mask) while respecting an additional boolean ``mask``.
    Positions that are masked or causally forbidden receive probability 0.

    Parameters
    ----------
    logits : np.ndarray
        Raw attention scores, shape (n, n).
    mask : np.ndarray
        Boolean array of the same shape; True means the key is valid.
    block_size : int
        Size of KV blocks.  It is unused in this reference implementation,
        but kept for API compatibility.

    Returns
    -------
    probs : np.ndarray
        Softmax probabilities, shape (n, n), dtype float64.
    """
    logits = np.asarray(logits, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)
    n = logits.shape[0]
    out = np.zeros_like(logits, dtype=np.float64)

    for i in range(n):
        has_valid = False
        max_val = 0.0
        for j in range(i + 1):
            if mask[i, j]:
                val = float(logits[i, j])
                if not has_valid:
                    max_val = val
                    has_valid = True
                elif val > max_val:
                    max_val = val

        if not has_valid:
            continue

        sum_exp = 0.0
        for j in range(i + 1):
            if mask[i, j]:
                sum_exp += math.exp(float(logits[i, j]) - max_val)

        for j in range(i + 1):
            if mask[i, j]:
                out[i, j] = math.exp(float(logits[i, j]) - max_val) / sum_exp

    return out
