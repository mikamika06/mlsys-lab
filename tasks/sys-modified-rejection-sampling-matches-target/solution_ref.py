import numpy as np


def rejection_sample_block(P: np.ndarray, Q: np.ndarray, n_draws: int, seed: int) -> np.ndarray:
    """
    For every row (verification position) k, independently repeat
    n_draws times: draw x ~ Q[k], accept x with probability
    min(1, P[k,x]/Q[k,x]), otherwise resample from the residual
    distribution normalize(max(P[k]-Q[k], 0)).

    P, Q: (K, V) float64 target/draft distributions (rows sum to 1).
    n_draws: number of independent repetitions per row.
    seed: seeds np.random.default_rng for all randomness used here.

    Returns E: (K, V) float64 empirical distribution of the resulting
    output token at each position (E[k] approximates P[k]).
    """
    P = np.asarray(P, dtype=np.float64)
    Q = np.asarray(Q, dtype=np.float64)
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
