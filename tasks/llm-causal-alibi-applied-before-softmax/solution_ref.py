import math
import numpy as np

def causal_alibi_logits(logits: np.ndarray, alibi_bias: np.ndarray) -> np.ndarray:
    """
    Compute attention probabilities with a causal mask and ALiBi bias.

    Parameters
    ----------
    logits : np.ndarray
        Raw attention logits of shape (seq_len, seq_len).
    alibi_bias : np.ndarray
        Linear bias to add to each logit, same shape as ``logits``.

    Returns
    -------
    probs : np.ndarray
        Row‑wise softmax probabilities after applying the causal mask and bias.
    """
    if logits.shape != alibi_bias.shape:
        raise ValueError("logits and alibi_bias must have identical shapes")

    rows, cols = logits.shape
    probs = np.zeros((rows, cols), dtype=np.float64)

    for i in range(rows):
        max_val = -float('inf')
        for j in range(cols):
            if j <= i:
                val = float(logits[i, j]) + float(alibi_bias[i, j])
            else:
                val = -float('inf')
            if val > max_val:
                max_val = val

        exp_vals = [0.0] * cols
        exp_sum = 0.0
        for j in range(cols):
            if j <= i:
                val = float(logits[i, j]) + float(alibi_bias[i, j])
                e = math.exp(val - max_val)
            else:
                e = 0.0
            exp_vals[j] = e
            exp_sum += e

        for j in range(cols):
            probs[i, j] = exp_vals[j] / exp_sum

    return probs
