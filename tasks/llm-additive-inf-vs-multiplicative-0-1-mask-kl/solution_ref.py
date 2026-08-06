import math
import numpy as np

def masked_softmax(logits: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Compute softmax probabilities with additive -inf masking.

    Parameters
    ----------
    logits : np.ndarray
        Raw attention scores of shape (batch, seq_len).
    mask : np.ndarray
        Boolean or integer mask of the same shape. Positions where mask==0 are masked out.

    Returns
    -------
    probs : np.ndarray
        Softmax probabilities with masked entries set to zero and each row summing to one.
    """
    batch, seq_len = logits.shape
    probs = np.empty((batch, seq_len), dtype=logits.dtype)

    for i in range(batch):
        max_val = -float('inf')
        for j in range(seq_len):
            val = float(logits[i, j]) if mask[i, j] else -float('inf')
            if val > max_val:
                max_val = val

        sum_exp = 0.0
        exp_vals = []
        for j in range(seq_len):
            if mask[i, j]:
                e = math.exp(float(logits[i, j]) - max_val)
            else:
                e = math.exp(-float('inf') - max_val)
            exp_vals.append(e)
            sum_exp += e

        for j in range(seq_len):
            probs[i, j] = exp_vals[j] / sum_exp

    return probs
