import math


def recompute_probs_from_lse(Q: list[list[float]], K: list[list[float]], lse: list[float]) -> list[list[float]]:
    """Recompute attention probabilities from Q, K and a stored per-row
    logsumexp, without ever computing a row max or normalizing by a row sum.

    P = exp(Q @ K.T / sqrt(d) - lse[:, None])
    """
    n = len(Q)
    d = len(Q[0])
    m = len(K)
    inv_sqrt_d = 1.0 / math.sqrt(d)

    out = [[0.0] * m for _ in range(n)]
    for i in range(n):
        for j in range(m):
            dot = 0.0
            for k in range(d):
                dot += Q[i][k] * K[j][k]
            score = dot * inv_sqrt_d
            out[i][j] = math.exp(score - lse[i])

    return out
