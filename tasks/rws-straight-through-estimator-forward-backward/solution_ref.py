import numpy as np


def _softmax(z):
    z = z - np.max(z, axis=-1, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=-1, keepdims=True)


def ste_argmax(logits, upstream_grad):
    """Straight-through estimator for a hard argmax selection.

    Many compression pipelines need a *hard*, non-differentiable choice in
    the forward pass (e.g. picking one codebook entry, one expert, one bit
    pattern) but still need a useful gradient to flow backward through that
    choice during training. The straight-through estimator (STE) does this
    by decoupling the two passes:

    - **Forward**: emit the hard one-hot vector of `argmax(logits)` per
      row (ties broken by the lowest index, matching `np.argmax`).
    - **Backward**: pretend the forward pass had instead been the *soft*
      `softmax(logits)`, and back-propagate the upstream gradient through
      that softmax's Jacobian:

        grad_logits = J_softmax^T @ upstream_grad

      where `J_softmax[i, j] = softmax_i * (delta_ij - softmax_j)`. In
      closed form, the vector-Jacobian product simplifies to

        grad_logits = softmax * (upstream_grad - sum(upstream_grad * softmax))

      (sum taken along the last axis, broadcasting back over it).

    Parameters
    ----------
    logits : np.ndarray, shape (..., C)
    upstream_grad : np.ndarray, shape (..., C)
        Gradient of the downstream loss with respect to the STE's forward
        output.

    Returns
    -------
    y_hard : np.ndarray, float64, shape (..., C)
        One-hot forward output.
    grad_logits : np.ndarray, float64, shape (..., C)
        Straight-through surrogate gradient with respect to `logits`.
    """
    logits = np.asarray(logits, dtype=np.float64)
    upstream_grad = np.asarray(upstream_grad, dtype=np.float64)

    idx = np.argmax(logits, axis=-1)
    y_hard = np.zeros_like(logits)
    np.put_along_axis(y_hard, idx[..., None], 1.0, axis=-1)

    y_soft = _softmax(logits)
    dot = np.sum(upstream_grad * y_soft, axis=-1, keepdims=True)
    grad_logits = y_soft * (upstream_grad - dot)

    return y_hard, grad_logits
