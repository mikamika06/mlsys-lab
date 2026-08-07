import random


def rejection_sample_block(P: list[list[float]], Q: list[list[float]], n_draws: int, seed: int) -> list[list[float]]:
    """
    For every row (verification position) k, independently repeat
    n_draws times: draw x ~ Q[k], accept x with probability
    min(1, P[k,x]/Q[k,x]), otherwise resample from the residual
    distribution normalize(max(P[k]-Q[k], 0)).

    P, Q: (K, V) target/draft distributions (rows sum to 1).
    n_draws: number of independent repetitions per row.
    seed: seeds random.Random for all randomness used here.

    Returns E: (K, V) empirical distribution of the resulting
    output token at each position (E[k] approximates P[k]).
    """
    rng = random.Random(seed)

    K = len(P)
    V = len(P[0])
    counts = [[0] * V for _ in range(K)]

    for k in range(K):
        p = P[k]
        q = Q[k]

        residual = [0.0] * V
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
            residual = list(q)

        tokens = rng.choices(range(V), weights=q, k=n_draws)
        u = [rng.random() for _ in range(n_draws)]

        accept_prob = [0.0] * n_draws
        accept = [False] * n_draws
        for i in range(n_draws):
            t = tokens[i]
            q_val = q[t]
            denom = q_val if q_val >= 1e-300 else 1e-300
            ratio = p[t] / denom
            ap = 1.0 if 1.0 < ratio else ratio
            accept_prob[i] = ap
            accept[i] = u[i] < ap

        out = [0] * n_draws
        n_reject = 0
        for i in range(n_draws):
            if accept[i]:
                out[i] = tokens[i]
            else:
                out[i] = -1
                n_reject += 1

        if n_reject > 0:
            resampled = rng.choices(range(V), weights=residual, k=n_reject)
            res_idx = 0
            for i in range(n_draws):
                if not accept[i]:
                    out[i] = resampled[res_idx]
                    res_idx += 1

        row_counts = [0] * V
        for i in range(n_draws):
            row_counts[out[i]] += 1
        counts[k] = row_counts

    return [[counts[k][v] / float(n_draws) for v in range(V)] for k in range(K)]
