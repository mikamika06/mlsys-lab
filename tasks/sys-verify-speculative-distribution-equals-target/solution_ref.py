import numpy as np


def speculative_distribution(draft_probs, target_probs, steps, seed):
    q = np.asarray(draft_probs, dtype=np.float64)
    p = np.asarray(target_probs, dtype=np.float64)
    rng = np.random.default_rng(seed)

    residual = np.maximum(p - q, 0.0)
    total = residual.sum()
    if total > 0:
        residual = residual / total

    counts = np.zeros(len(p), dtype=np.int64)

    for _ in range(steps):
        token = int(rng.choice(len(q), p=q))
        if q[token] == 0:
            accept = 0.0
        else:
            accept = min(1.0, p[token] / q[token])

        if rng.random() < accept:
            out = token
        else:
            out = int(rng.choice(len(q), p=residual))

        counts[out] += 1

    return counts.astype(np.float64) / float(steps)
