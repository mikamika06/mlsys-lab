import random


def speculative_distribution(
    draft_probs: list[float],
    target_probs: list[float],
    steps: int,
    seed: int,
) -> list[float]:
    q = list(draft_probs)
    p = list(target_probs)
    rng = random.Random(seed)

    n = len(p)
    residual = []
    for i in range(n):
        diff = p[i] - q[i]
        residual.append(diff if diff > 0.0 else 0.0)

    total = 0.0
    for i in range(n):
        total += residual[i]

    if total > 0:
        for i in range(n):
            residual[i] /= total

    counts = [0] * n

    for _ in range(steps):
        token = rng.choices(range(len(q)), weights=q, k=1)[0]
        if q[token] == 0:
            accept = 0.0
        else:
            accept = min(1.0, p[token] / q[token])

        if rng.random() < accept:
            out = token
        else:
            out = rng.choices(range(len(q)), weights=residual, k=1)[0]

        counts[out] += 1

    ret = [0.0] * n
    for i in range(n):
        ret[i] = counts[i] / float(steps)
    return ret
