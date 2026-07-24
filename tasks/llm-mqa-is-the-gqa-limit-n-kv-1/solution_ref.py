import numpy as np

def gqa_limit_nkv_1(Q: np.ndarray, K: np.ndarray, V: np.ndarray) -> np.ndarray:
    """
    Compute multi‑query attention (GQA with n_kv=1).

    Parameters
    ----------
    Q : np.ndarray
        Queries of shape (B, N_q, d_k).
    K : np.ndarray
        Keys of shape (B, N_k, d_k).
    V : np.ndarray
        Values of shape (B, N_v, d_v).

    Returns
    -------
    np.ndarray
        Attention output of shape (B, N_q, d_v).
    """
    scores = Q @ K.transpose(0, 2, 1) / np.sqrt(K.shape[-1])
    # softmax over the last axis (keys dimension)
    e = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
    weights = e / np.sum(e, axis=-1, keepdims=True)
    return weights @ V
