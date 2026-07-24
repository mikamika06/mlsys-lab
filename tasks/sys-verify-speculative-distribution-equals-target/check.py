import numpy as np


def _oracle(draft_probs, target_probs, steps, seed):
    rng = np.random.default_rng(seed)
    q = np.asarray(draft_probs, dtype=np.float64)
    p = np.asarray(target_probs, dtype=np.float64)
    counts = np.zeros_like(p, dtype=np.int64)

    residual = np.maximum(p - q, 0.0)
    residual_sum = residual.sum()
    if residual_sum > 0:
        residual = residual / residual_sum

    for _ in range(steps):
        token = int(rng.choice(len(q), p=q))
        accept_prob = 1.0 if q[token] == 0 else min(1.0, p[token] / q[token])
        if rng.random() < accept_prob:
            out = token
        else:
            out = int(rng.choice(len(q), p=residual))
        counts[out] += 1

    return counts.astype(np.float64) / steps


def _kl(p, q):
    return float(np.sum(p * (np.log(p + 1e-12) - np.log(q + 1e-12))))


def grade(sol, fx) -> dict:
    cases = [
        (
            np.array([0.70, 0.20, 0.10]),
            np.array([0.40, 0.35, 0.25]),
            100000,
            7,
        ),
        (
            np.array([0.15, 0.55, 0.20, 0.10]),
            np.array([0.30, 0.25, 0.25, 0.20]),
            100000,
            19,
        ),
        (
            np.array([0.05, 0.05, 0.15, 0.25, 0.50]),
            np.array([0.10, 0.20, 0.20, 0.30, 0.20]),
            120000,
            31,
        ),
    ]

    values = []
    for draft, target, steps, seed in cases:
        try:
            got = np.asarray(
                sol.speculative_distribution(
                    draft.copy(), target.copy(), steps, seed
                ),
                dtype=np.float64,
            )
        except Exception:
            return {"mean_kl": 1.0}

        if got.shape != target.shape:
            return {"mean_kl": 1.0}

        values.append(_kl(target, got))

    return {"mean_kl": float(np.mean(values))}
