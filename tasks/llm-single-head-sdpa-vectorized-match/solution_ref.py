import numpy as np

def sdpa_single_head(Q: np.ndarray, K: np.ndarray, V: np.ndarray) -> np.ndarray:
    Q = np.asarray(Q, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    d_head = Q.shape[1]
    scores = Q @ K.T / np.sqrt(d_head)
    e = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
    softmax = e / np.sum(e, axis=-1, keepdims=True)
    return softmax @ V
