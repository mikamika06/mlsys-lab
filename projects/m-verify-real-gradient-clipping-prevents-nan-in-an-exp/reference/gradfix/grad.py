import numpy as np

def clip_gradients(grads, max_norm):
    total_norm = np.sqrt(sum(np.sum(g ** 2) for g in grads))
    if total_norm > max_norm:
        clip_coef = max_norm / (total_norm + 1e-6)
        return [g * clip_coef for g in grads], total_norm
    return [g.copy() for g in grads], total_norm

def simulate_training(initial_weights, grad_sequences, max_norm, lr):
    weights = initial_weights.copy()
    for grads in grad_sequences:
        grads_clipped, _ = clip_gradients(grads, max_norm)
        weights = weights - lr * grads_clipped
        if not np.isfinite(weights).all():
            return True, weights
    return False, weights
