import numpy as np
from gradclip.clipping import clip_grad_norm


def run_training_step(weights, grad, max_norm, lr):
    clipped, _ = clip_grad_norm([grad], max_norm)
    new_weights = weights - lr * clipped[0]
    has_nan = bool(np.isnan(new_weights).any() or np.isinf(new_weights).any())
    return new_weights, has_nan
