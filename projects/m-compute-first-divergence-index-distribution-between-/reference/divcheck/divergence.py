import numpy as np


def compute_divergence_distribution(set_a, set_b):
    a = np.asarray(set_a, dtype=np.int64)
    b = np.asarray(set_b, dtype=np.int64)
    if a.shape != b.shape:
        raise ValueError("Sequence sets must have identical dimensions.")

    n_prompts, seq_len = a.shape
    mismatches = (a != b)

    counts = np.zeros(seq_len + 1, dtype=np.int64)
    for i in range(n_prompts):
        diff = np.where(mismatches[i])[0]
        if len(diff) == 0:
            counts[seq_len] += 1
        else:
            counts[diff[0]] += 1

    return counts.tolist()
