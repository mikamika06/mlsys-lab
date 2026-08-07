import numpy as np


def clip_grad_norm(grads, max_norm):
    total_norm = np.sqrt(sum(np.sum(np.square(g)) for g in grads))
    if total_norm > max_norm:
        scale = max_norm / (total_norm + 1e-6)
        clipped = [g * scale for g in grads]
    else:
        clipped = [g.copy() for g in grads]
    return clipped, total_norm


def simulate_nf4_cycles(tensor, steps):
    levels = np.array([-1.0, -0.6961928, -0.52507305, -0.39491749, -0.28444138, -0.18477343, -0.09105004, 0.0, 0.0795800, 0.1609302, 0.2479011, 0.3478914, 0.4679901, 0.6106249, 0.8160759, 1.0], dtype=np.float32)
    current = tensor.copy()
    for _ in range(steps):
        flat = current.ravel()
        indices = np.abs(flat[:, None] - levels[None, :]).argmin(axis=1)
        quantized = levels[indices].reshape(current.shape)
        current = quantized
    error = np.mean(np.square(current - tensor))
    return current, error


def run_training_step(weights, grad, max_norm, lr):
    clipped, _ = clip_grad_norm([grad], max_norm)
    new_weights = weights - lr * clipped[0]
    has_nan = bool(np.isnan(new_weights).any() or np.isinf(new_weights).any())
    return new_weights, has_nan
