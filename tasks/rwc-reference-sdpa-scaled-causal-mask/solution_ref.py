import numpy as np

def scaled_dot_product_attention(
    Q: np.ndarray,
    K: np.ndarray,
    V: np.ndarray,
    *,
    causal: bool = False
) -> np.ndarray:
    """
    Compute scaled dot‑product attention with optional causal masking.

    Parameters
    ----------
    Q, K : np.ndarray
        Query and key tensors of shape (B, N, d_k).
    V : np.ndarray
        Value tensor of shape (B, N, d_v).
    causal : bool, default False
        If True, apply a lower‑triangular causal mask.

    Returns
    -------
    out : np.ndarray
        Attention output of shape (B, N, d_v) with dtype float64.
    """
    Q = np.asarray(Q, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)

    d_k = Q.shape[-1]
    scores = Q @ K.transpose(0, 2, 1) / np.sqrt(d_k)

    if causal:
        seq_len = scores.shape[1]
        mask = np.triu(np.full((seq_len, seq_len), -np.inf), k=1)
        scores += mask

    # Softmax along the last axis
    exp_scores = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
    attn_weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)

    return attn_weights @ V
