import numpy as np

def mqa_single_kv_broadcast(Q: np.ndarray,
                            K: np.ndarray,
                            V: np.ndarray) -> np.ndarray:
    """
    Correct implementation of MQA with a single KV head.
    
    Parameters
    ----------
    Q : ndarray, shape (n_q, h, d_k)
        Query matrix for all heads.
    K : ndarray, shape (1, d_k)
        Shared key vector.
    V : ndarray, shape (1, d_v)
        Shared value vector.

    Returns
    -------
    out : ndarray, shape (n_q, h, d_v)
        Attention output broadcasted across all queries and heads.
    """
    d_k = Q.shape[-1]
    # Compute scores for each head: (n_q, h)
    scores = np.einsum('qhd,hd->qh', Q, K) / np.sqrt(d_k)

    # Softmax over the single key dimension
    weights = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
    weights /= np.sum(weights, axis=-1, keepdims=True)

    # Broadcast V across heads and queries: (n_q, h, d_v)
    out = weights[..., None] * V[0]
    return out
