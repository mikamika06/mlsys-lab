import numpy as np


def speculative_distribution(draft_probs, target_probs, steps, seed):
    q = np.asarray(draft_probs, dtype=np.float64)
    p = np.asarray(target_probs, dtype=np.float64)
    rng = np.random.default_rng(seed)

    n = len(p)
    residual_list = []
    for i in range(n):
        diff = p[i] - q[i]
        residual_list.append(diff if diff > 0.0 else 0.0)
    residual = np.asarray(residual_list, dtype=np.float64)

    total = 0.0
    for i in range(n):
        total += residual[i]

    if total > 0:
        for i in range(n):
            residual[i] /= total

    counts = np.zeros(n, dtype=np.int64)

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

    ret = np.zeros(n, dtype=np.float64)
    for i in range(n):
        ret[i] = counts[i] / float(steps)
    return ret
