import math
import numpy as np


def cosine_embedding_loss_and_grad(h_t: np.ndarray, h_s: np.ndarray, eps: float = 1e-8):
    """DistilBERT's hidden-state cosine embedding loss (target=1) and its
    gradient w.r.t. h_s. See task.md for the derivation.
    """
    h_t = np.asarray(h_t, dtype=np.float64)
    h_s = np.asarray(h_s, dtype=np.float64)
    B = h_t.shape[0]
    D = h_t.shape[1]

    na = [0.0] * B
    nb = [0.0] * B
    dot = [0.0] * B
    denom = [0.0] * B
    cos = [0.0] * B

    for i in range(B):
        sum_a = 0.0
        sum_b = 0.0
        sum_dot = 0.0
        for j in range(D):
            sum_a += h_t[i, j] * h_t[i, j]
            sum_b += h_s[i, j] * h_s[i, j]
            sum_dot += h_t[i, j] * h_s[i, j]
        na[i] = math.sqrt(sum_a)
        nb[i] = math.sqrt(sum_b)
        dot[i] = sum_dot
        denom[i] = na[i] * nb[i] + eps
        cos[i] = dot[i] / denom[i]

    loss_sum = 0.0
    for i in range(B):
        loss_sum += 1.0 - cos[i]
    loss = float(loss_sum / B)

    grad_list = []
    for i in range(B):
        row_grad = []
        term1_denom = denom[i]
        coeff2 = (dot[i] * na[i]) / (nb[i] * (term1_denom * term1_denom))
        for j in range(D):
            term1_val = h_t[i, j] / term1_denom
            term2_val = coeff2 * h_s[i, j]
            val = -(term1_val - term2_val) / B
            row_grad.append(val)
        grad_list.append(row_grad)

    grad = np.asarray(grad_list, dtype=np.float64)

    return loss, grad
