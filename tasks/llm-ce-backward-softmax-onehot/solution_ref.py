import math
import numpy as np


def ce_backward(logits: np.ndarray, labels: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    n_samples, n_classes = logits.shape
    probs = np.empty((n_samples, n_classes), dtype=np.float64)

    for i in range(n_samples):
        max_val = logits[i, 0]
        for j in range(1, n_classes):
            if logits[i, j] > max_val:
                max_val = logits[i, j]

        sum_exp = 0.0
        for j in range(n_classes):
            e = math.exp(logits[i, j] - max_val)
            probs[i, j] = e
            sum_exp += e

        for j in range(n_classes):
            probs[i, j] = probs[i, j] / sum_exp

        label = labels[i]
        probs[i, label] -= 1.0

        for j in range(n_classes):
            probs[i, j] /= n_samples

    return probs
