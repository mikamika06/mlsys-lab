import numpy as np


def cosine_embedding_loss_and_grad(h_t: np.ndarray, h_s: np.ndarray, eps: float = 1e-8):
    """DistilBERT's hidden-state cosine embedding loss (target=1) and its
    gradient w.r.t. h_s.

    h_t: (B, d) float64 teacher hidden states (constant).
    h_s: (B, d) float64 student hidden states (differentiate w.r.t. this).
    eps: small constant added to the denominator for numerical stability.

    For each row i: na = ||h_t[i]||, nb = ||h_s[i]||, dot = h_t[i].h_s[i],
    denom = na*nb + eps, cos = dot/denom, loss_i = 1 - cos.
    loss = mean(loss_i) over the batch.
    grad[i] = -(h_t[i]/denom - (dot*na/(nb*denom**2)) * h_s[i]) / B

    Returns (loss, grad): loss a float, grad a (B, d) float64 array
    (dL/d h_s).
    """
    raise NotImplementedError('your code here')
