import numpy as np

def mha_forward(X: np.ndarray,
                Wq: np.ndarray,
                Wk: np.ndarray,
                Wv: np.ndarray,
                Wo: np.ndarray) -> np.ndarray:
    """
    Full multi‑head attention forward pass.

    Parameters
    ----------
    X : np.ndarray, shape (B, T, d_model)
        Input sequence.
    Wq, Wk, Wv, Wo : np.ndarray, shape (d_model, d_model)
        Projection matrices.  The hidden dimension is split into H=4 heads.
    Returns
    -------
    Y : np.ndarray, shape (B, T, d_model)
        Output of the multi‑head attention layer.
    """
    batch, seq_len, d_model = X.shape
    H = 4
    head_dim = d_model // H
    assert d_model % H == 0

    # Linear projections
    Q = X @ Wq
    K = X @ Wk
    V = X @ Wv

    # Reshape to heads: (B, T, H, h)
    Qh = Q.reshape(batch, seq_len, H, head_dim).transpose(0, 2, 1, 3)
    Kh = K.reshape(batch, seq_len, H, head_dim).transpose(0, 2, 1, 3)
    Vh = V.reshape(batch, seq_len, H, head_dim).transpose(0, 2, 1, 3)

    # Scaled dot‑product attention
    scores = Qh @ Kh.transpose(0, 1, 3, 2) / np.sqrt(head_dim)
    attn = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
    attn /= np.sum(attn, axis=-1, keepdims=True)

    # Weighted sum of values
    out_h = attn @ Vh

    # Merge heads and final projection
    out = out_h.transpose(0, 2, 1, 3).reshape(batch, seq_len, d_model)
    return out @ Wo
