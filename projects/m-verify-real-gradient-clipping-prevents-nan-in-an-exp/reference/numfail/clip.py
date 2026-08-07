import numpy as np


def clip_grad_norm(grads, max_norm):
    total_sq = sum(np.sum(np.square(g)) for g in grads)
    total_norm = float(np.sqrt(total_sq))
    if max_norm is not None and max_norm > 0 and total_norm > max_norm:
        scale = max_norm / (total_norm + 1e-12)
        clipped = [g * scale for g in grads]
    else:
        clipped = [g.copy() for g in grads]
    return clipped, total_norm


def run_exploding_loop(init_w, lr, steps, max_norm=None):
    w = np.array(init_w, dtype=np.float64).copy()
    history = {"weights": [], "grads": [], "has_nan": False}
    for _ in range(steps):
        grad = np.exp(w)
        if np.isnan(grad).any() or np.isinf(grad).any():
            history["has_nan"] = True
            break
        if max_norm is not None:
            [clipped_grad], norm = clip_grad_norm([grad], max_norm)
        else:
            clipped_grad = grad.copy()
        w = w + lr * clipped_grad
        if np.isnan(w).any() or np.isinf(w).any():
            history["has_nan"] = True
            break
        history["weights"].append(w.copy())
        history["grads"].append(clipped_grad.copy())
    return history
