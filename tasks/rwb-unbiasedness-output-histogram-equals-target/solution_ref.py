import numpy as np


def speculative_histogram(p, q, seed, n_samples):
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)

    length = len(p)
    residual_mass = np.zeros(length, dtype=np.float64)
    for i in range(length):
        diff = p[i] - q[i]
        residual_mass[i] = diff if diff > 0.0 else 0.0

    residual_total = 0.0
    for i in range(length):
        residual_total += residual_mass[i]

    residual = np.zeros(length, dtype=np.float64)
    if residual_total > 0:
        for i in range(length):
            residual[i] = residual_mass[i] / residual_total
    else:
        for i in range(length):
            residual[i] = p[i]

    rng = np.random.default_rng(seed)
    counts = np.zeros(length, dtype=np.int64)

    accept_prob = np.zeros(length, dtype=np.float64)
    for i in range(length):
        if q[i] > 0:
            val = p[i] / q[i]
            accept_prob[i] = 1.0 if val > 1.0 else val
        else:
            accept_prob[i] = 0.0

    for _ in range(n_samples):
        proposal = int(rng.choice(len(q), p=q))
        if rng.random() < accept_prob[proposal]:
            token = proposal
        else:
            token = int(rng.choice(len(p), p=residual))
        counts[token] += 1

    result = np.zeros(length, dtype=np.float64)
    for i in range(length):
        result[i] = counts[i] / n_samples

    return result
