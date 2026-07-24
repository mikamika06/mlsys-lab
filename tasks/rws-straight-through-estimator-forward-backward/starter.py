import numpy as np


def ste_argmax(logits, upstream_grad):
    """Straight-through estimator for a hard argmax selection.

    logits, upstream_grad: arrays of shape (..., C).

    Forward: y_hard = one-hot(argmax(logits, axis=-1)) (ties broken by the
    lowest index, matching np.argmax).

    Backward (STE): pretend the forward pass had been softmax(logits)
    instead, and back-propagate upstream_grad through that softmax's
    Jacobian:

        grad_logits = softmax * (upstream_grad - sum(upstream_grad * softmax))

    (sum over the last axis, broadcast back over it).

    Returns (y_hard, grad_logits), both float64 arrays with the same shape
    as logits.
    """
    raise NotImplementedError('your code here')
