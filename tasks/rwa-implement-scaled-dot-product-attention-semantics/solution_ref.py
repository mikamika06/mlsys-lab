import numpy as np

def scaled_dot_product_attention(
    Q: np.ndarray,
    K: np.ndarray,
    V: np.ndarray,
    mask: np.ndarray | None = None,
    causal: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute scaled dot‑product attention with optional masking.

    Parameters
    ----------
    Q : (..., T_q, d_k)
        Query tensor.
    K : (..., T_k, d_k)
        Key tensor.
    V : (..., T_k, d_v)
        Value tensor.
    mask : array_like or None, optional
        Broadcastable to the logits shape. If bool, positions with False are masked out.
        If float, values are added element‑wise to the logits before softmax.
    causal : bool, default False
        Whether to apply a causal (triangular) mask.

    Returns
    -------
    output : (..., T_q, d_v)
        Aggregated values.
    weights : (..., T_q, T_k)
        Attention probabilities.
    """
    d_k = K.shape[-1]
    scale = 1 / np.sqrt(d_k)

    logits = np.matmul(Q, K.swapaxes(-2, -1)) * scale

    if causal:
        seq_q, seq_k = logits.shape[-2], logits.shape[-1]
        causal_mask = np.triu(np.full((seq_q, seq_k), -np.inf), 1)
        logits += causal_mask

    if mask is not None:
        if mask.dtype == bool:
            logits[~mask] = -np.inf
        else:
            logits += mask

    # stable softmax
    max_logits = np.max(logits, axis=-1, keepdims=True)
    exp = np.exp(logits - max_logits)
    weights = exp / np.sum(exp, axis=-1, keepdims=True)

    output = np.matmul(weights, V)
    return output, weights
