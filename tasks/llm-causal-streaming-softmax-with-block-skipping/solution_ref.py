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

    # Causal mask: lower triangular including diagonal
    causal_mask = np.tril(np.ones((n, n), dtype=bool))
    combined_mask = mask & causal_mask

    for i in range(n):
        row_mask = combined_mask[i]
        if not row_mask.any():
            continue
        vals = logits[i, row_mask]
        max_val = np.max(vals)
        exp_vals = np.exp(vals - max_val)
        probs = exp_vals / np.sum(exp_vals)
        out[i, row_mask] = probs

    return out
