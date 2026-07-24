import numpy as np


def alibi_extrapolation_metric(num_heads, trained_len, extra_len):
    slopes = 2.0 ** (-(np.arange(num_heads, dtype=np.float64) + 1.0) / num_heads)
    total = 0.0
    count = 0
    for q in range(trained_len, trained_len + extra_len):
        distances = np.arange(q, -1, -1, dtype=np.float64)
        for slope in slopes:
            logits = -slope * distances
            logits -= np.max(logits)
            weights = np.exp(logits)
            weights /= np.sum(weights)
            total += np.sum(weights * distances) / (q + 1.0)
            count += 1
    return float(total / count)
