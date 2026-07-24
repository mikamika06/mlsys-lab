import numpy as np

def sdpa(Q: np.ndarray, K: np.ndarray, V: np.ndarray) -> np.ndarray:
    d_k = Q.shape[-1]
    logits = (Q @ K.T) / np.sqrt(d_k)
    exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
    softmax = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
    out = softmax @ V
    return out.astype(Q.dtype)
