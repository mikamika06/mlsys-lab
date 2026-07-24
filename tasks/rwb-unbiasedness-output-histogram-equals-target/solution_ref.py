import numpy as np


def speculative_histogram(p, q, seed, n_samples):
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)

    residual_mass = np.maximum(p - q, 0.0)
    residual_total = float(np.sum(residual_mass))
    if residual_total > 0:
        residual = residual_mass / residual_total
    else:
        residual = p.copy()

    rng = np.random.default_rng(seed)
    counts = np.zeros(p.shape[0], dtype=np.int64)

    accept_prob = np.minimum(
        1.0,
        np.divide(p, q, out=np.zeros_like(p), where=q > 0),
    )

    for _ in range(n_samples):
        proposal = int(rng.choice(len(q), p=q))
        if rng.random() < accept_prob[proposal]:
            token = proposal
        else:
            token = int(rng.choice(len(p), p=residual))
        counts[token] += 1

    return counts.astype(np.float64) / n_samples
