import numpy as np


def compare_2_4_masks(weights, logits):
    weights = np.asarray(weights, dtype=np.float64)
    logits = np.asarray(logits, dtype=np.float64)

    soft_idx = np.argsort(-logits, axis=1, kind="stable")[:, :2]
    mag_idx = np.argsort(-np.abs(weights), axis=1, kind="stable")[:, :2]

    soft_mask = np.zeros_like(weights)
    mag_mask = np.zeros_like(weights)

    rows = np.arange(weights.shape[0])
    soft_mask[rows[:, None], soft_idx] = 1.0
    mag_mask[rows[:, None], mag_idx] = 1.0

    soft_retained = np.sum(np.abs(weights) * soft_mask, axis=1)
    magnitude_retained = np.sum(np.abs(weights) * mag_mask, axis=1)

    soft_error = np.sum((weights - weights * soft_mask) ** 2, axis=1)
    magnitude_error = np.sum((weights - weights * mag_mask) ** 2, axis=1)

    soft_total = np.sum(soft_error)
    magnitude_total = np.sum(magnitude_error)

    if soft_total < magnitude_total:
        better = "soft"
    elif magnitude_total < soft_total:
        better = "magnitude"
    else:
        better = "tie"

    return {
        "soft_retained": soft_retained,
        "magnitude_retained": magnitude_retained,
        "soft_error": soft_error,
        "magnitude_error": magnitude_error,
        "better": better,
    }
