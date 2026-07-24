import numpy as np

def optimizer_step(params, grads, scale, lr):
    """Apply gradient descent step.

    BUG: grads are scaled by `scale` (AMP loss scaling).
    You must unscale them before applying the update.

    params: list of np.ndarray (model weights)
    grads:  list of np.ndarray (scaled gradients = true_grad * scale)
    scale:  float, the AMP loss scale factor
    lr:     float, learning rate
    Returns: updated params list
    """
    for i, (p, g) in enumerate(zip(params, grads)):
        params[i] = p - lr * g   # BUG: missing / scale
    return params
