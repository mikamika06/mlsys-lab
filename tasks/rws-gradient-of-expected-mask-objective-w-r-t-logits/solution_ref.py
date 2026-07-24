import numpy as np


def expected_mask_grad(logits, values, target):
    logits = np.asarray(logits, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)

    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp_logits = np.exp(shifted)
    probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

    mask = probs @ values
    d_loss_d_mask = 2.0 * (mask - target)

    centered_values = values[None, :] - mask[:, None]
    return (d_loss_d_mask[:, None] * probs * centered_values).astype(np.float64)
