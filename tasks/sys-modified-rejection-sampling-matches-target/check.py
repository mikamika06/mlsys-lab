import numpy as np


def _oracle(P, Q, n_draws, seed):
    rng = np.random.default_rng(seed)
    K, V = P.shape
    counts = np.zeros((K, V), dtype=np.int64)

    for k in range(K):
        p = P[k]
        q = Q[k]
        residual = np.maximum(p - q, 0.0)
        rsum = residual.sum()
        residual = residual / rsum if rsum > 0 else q.copy()

        tokens = rng.choice(V, size=n_draws, p=q)
        u = rng.random(n_draws)
        accept_prob = np.minimum(1.0, p[tokens] / np.maximum(q[tokens], 1e-300))
        accept = u < accept_prob

        out = np.where(accept, tokens, -1)
        n_reject = int(np.sum(~accept))
        if n_reject > 0:
            resampled = rng.choice(V, size=n_reject, p=residual)
            out[~accept] = resampled

        counts[k] = np.bincount(out, minlength=V)

    return counts.astype(np.float64) / n_draws


def _kl_rows(p, freq, eps=1e-9):
    return np.sum(p * (np.log(p + eps) - np.log(freq + eps)), axis=-1)


def _cases():
    return [
        (
            np.array([[0.70, 0.20, 0.10],
                      [0.10, 0.10, 0.80],
                      [0.34, 0.33, 0.33]]),
            np.array([[0.40, 0.35, 0.25],
                      [0.50, 0.30, 0.20],
                      [0.20, 0.40, 0.40]]),
            60_000,
            7,
        ),
        (
            np.array([[0.15, 0.55, 0.20, 0.10],
                      [0.40, 0.10, 0.10, 0.40]]),
            np.array([[0.30, 0.25, 0.25, 0.20],
                      [0.25, 0.25, 0.25, 0.25]]),
            60_000,
            19,
        ),
        (
            np.array([[0.05, 0.05, 0.15, 0.25, 0.50]]),
            np.array([[0.10, 0.20, 0.20, 0.30, 0.20]]),
            80_000,
            31,
        ),
    ]


def grade(sol, fx) -> dict:
    kls = []
    for P, Q, n_draws, seed in _cases():
        try:
            got = np.asarray(
                sol.rejection_sample_block(P.copy(), Q.copy(), n_draws, seed),
                dtype=np.float64,
            )
        except Exception:
            return {"mean_kl": 999.0}

        if got.shape != P.shape:
            return {"mean_kl": 999.0}
        if not np.all(np.isfinite(got)):
            return {"mean_kl": 999.0}
        row_sums = got.sum(axis=-1)
        if not np.allclose(row_sums, 1.0, atol=1e-6):
            return {"mean_kl": 999.0}

        kls.extend(_kl_rows(P, got).tolist())

    return {"mean_kl": float(np.mean(kls))}
