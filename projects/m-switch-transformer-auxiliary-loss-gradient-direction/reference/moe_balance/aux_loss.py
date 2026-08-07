import numpy as np


def compute_switch_aux_loss(logits, alpha=0.01):
    """
    Computes Switch Transformer aux loss and its analytical gradient wrt logits.
    logits: shape (T, N) where T = num_tokens, N = num_experts.
    """
    T, N = logits.shape
    exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
    P = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
    P_mean = np.mean(P, axis=0)

    top1_indices = np.argmax(logits, axis=-1)
    f = np.zeros(N, dtype=np.float64)
    for idx in top1_indices:
        f[idx] += 1.0
    f /= T

    loss = alpha * N * np.sum(P_mean * f)

    # dL/dP_i = (alpha * N / T) * f_i
    # dP_{t,i} / dlogits_{t,j} = P_{t,i} * (delta_{ij} - P_{t,j})
    # dL / dlogits_{t,j} = sum_i (dL/dP_i) * P_{t,i} * (delta_{ij} - P_{t,j})
    #                    = (alpha * N / T) * P_{t,j} * (f_j - sum_i f_i * P_{t,i})
    factor = (alpha * N) / T
    dot_f_P = np.sum(P * f, axis=-1, keepdims=True)
    grad = factor * P * (f - dot_f_P)

    return float(loss), grad


def switch_grad_direction(logits, alpha=0.01):
    _, grad = compute_switch_aux_loss(logits, alpha=alpha)
    return grad
