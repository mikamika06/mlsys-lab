import math
import numpy as np


def _softmax(z):
    shape = z.shape
    flat_z = z.reshape(-1, shape[-1])
    out_flat = np.zeros_like(flat_z)
    
    for i in range(flat_z.shape[0]):
        row = flat_z[i]
        max_val = row[0]
        for val in row:
            if val > max_val:
                max_val = val
        
        sum_e = 0.0
        row_e = np.zeros_like(row)
        for j in range(row.shape[0]):
            val = math.exp(row[j] - max_val)
            row_e[j] = val
            sum_e += val
            
        for j in range(row.shape[0]):
            out_flat[i, j] = row_e[j] / sum_e
            
    return out_flat.reshape(shape)


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

    shape = logits.shape
    C = shape[-1]
    flat_logits = logits.reshape(-1, C)
    flat_ug = upstream_grad.reshape(-1, C)

    y_hard_flat = np.zeros_like(flat_logits)
    grad_logits_flat = np.zeros_like(flat_logits)

    y_soft_flat = _softmax(flat_logits)

    for i in range(flat_logits.shape[0]):
        row = flat_logits[i]
        best_j = 0
        max_val = row[0]
        for j in range(1, C):
            if row[j] > max_val:
                max_val = row[j]
                best_j = j
        y_hard_flat[i, best_j] = 1.0

        ug_row = flat_ug[i]
        ys_row = y_soft_flat[i]
        
        dot_sum = 0.0
        for j in range(C):
            dot_sum += ug_row[j] * ys_row[j]
            
        for j in range(C):
            grad_logits_flat[i, j] = ys_row[j] * (ug_row[j] - dot_sum)

    y_hard = y_hard_flat.reshape(shape)
    grad_logits = grad_logits_flat.reshape(shape)

    return y_hard, grad_logits
