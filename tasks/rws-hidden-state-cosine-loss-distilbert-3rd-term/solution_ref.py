import numpy as np


def cosine_embedding_loss_and_grad(h_t: np.ndarray, h_s: np.ndarray, eps: float = 1e-8):
    """DistilBERT's hidden-state cosine embedding loss (target=1) and its
    gradient w.r.t. h_s. See task.md for the derivation.
    """
    h_t = np.asarray(h_t, dtype=np.float64)
    h_s = np.asarray(h_s, dtype=np.float64)
    B = h_t.shape[0]

    na = np.linalg.norm(h_t, axis=1)
    nb = np.linalg.norm(h_s, axis=1)
    dot = np.sum(h_t * h_s, axis=1)
    denom = na * nb + eps
    cos = dot / denom
    loss = float(np.mean(1.0 - cos))

    term1 = h_t / denom[:, None]
    term2 = (dot * na / (nb * denom ** 2))[:, None] * h_s
    grad = -(term1 - term2) / B

    return loss, grad
