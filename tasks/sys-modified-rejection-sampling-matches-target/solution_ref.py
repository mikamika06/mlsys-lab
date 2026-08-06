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

        residual = np.empty(V, dtype=np.float64)
        rsum = 0.0
        for v in range(V):
            diff = p[v] - q[v]
            val = diff if diff > 0.0 else 0.0
            residual[v] = val
            rsum += val

        if rsum > 0.0:
            for v in range(V):
                residual[v] /= rsum
        else:
            residual = q.copy()

        tokens = rng.choice(V, size=n_draws, p=q)
        u = rng.random(n_draws)

        accept_prob = np.empty(n_draws, dtype=np.float64)
        accept = np.empty(n_draws, dtype=bool)
        for i in range(n_draws):
            t = tokens[i]
            q_val = q[t]
            denom = q_val if q_val >= 1e-300 else 1e-300
            ratio = p[t] / denom
            ap = 1.0 if 1.0 < ratio else ratio
            accept_prob[i] = ap
            accept[i] = u[i] < ap

        out = np.empty(n_draws, dtype=np.int64)
        n_reject = 0
        for i in range(n_draws):
            if accept[i]:
                out[i] = tokens[i]
            else:
                out[i] = -1
                n_reject += 1

        if n_reject > 0:
            resampled = rng.choice(V, size=n_reject, p=residual)
            res_idx = 0
            for i in range(n_draws):
                if not accept[i]:
                    out[i] = resampled[res_idx]
                    res_idx += 1

        row_counts = np.zeros(V, dtype=np.int64)
        for i in range(n_draws):
            row_counts[out[i]] += 1
        counts[k] = row_counts

    return counts.astype(np.float64) / n_draws
