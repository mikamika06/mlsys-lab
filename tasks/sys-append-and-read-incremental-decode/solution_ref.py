import numpy as np

def incremental_decode(embeddings: np.ndarray,
                       Wq: np.ndarray,
                       Wk: np.ndarray,
                       Wv: np.ndarray) -> np.ndarray:
    """
    Correct implementation of incremental KV‑cache decoding.

    Parameters
    ----------
    embeddings : np.ndarray, shape (n, d_in)
        Token embeddings.
    Wq, Wk, Wv : np.ndarray
        Weight matrices for Q, K and V respectively.

    Returns
    -------
    outputs : np.ndarray, shape (n, d_v)
        Incremental attention outputs.
    """
    n = embeddings.shape[0]
    d_k = Wk.shape[1]
    sqrt_dk = np.sqrt(d_k)

    # Cache for keys and values
    cache_K = np.empty((0, d_k), dtype=np.float64)
    cache_V = np.empty((0, Wv.shape[1]), dtype=np.float64)

    outputs = np.empty((n, Wv.shape[1]), dtype=np.float64)

    for t in range(n):
        x_t = embeddings[t]
        Q_t = x_t @ Wq
        K_t = x_t @ Wk
        V_t = x_t @ Wv

        # Append to cache
        cache_K = np.vstack((cache_K, K_t[None, :]))
        cache_V = np.vstack((cache_V, V_t[None, :]))

        scores = (Q_t @ cache_K.T) / sqrt_dk  # shape (t+1,)
        max_score = scores.max()
        exp_scores = np.exp(scores - max_score)
        alphas = exp_scores / exp_scores.sum()

        outputs[t] = alphas @ cache_V

    return outputs
