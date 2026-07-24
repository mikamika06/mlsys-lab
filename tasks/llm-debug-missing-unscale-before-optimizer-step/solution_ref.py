import numpy as np

def optimizer_step(params, grads, scale, lr):
    """Apply gradient descent with properly unscaled gradients.

    w <- w - lr * (g_scaled / scale)
    """
    for i, (p, g) in enumerate(zip(params, grads)):
        params[i] = p - lr * (g / scale)  # unscale before applying
    return params
