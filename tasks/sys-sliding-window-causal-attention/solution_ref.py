import math


def sliding_window_attention_tiled(Q: list[list[float]], K: list[list[float]], V: list[list[float]], window: int, block_size: int) -> list[list[float]]:
    """
    Sliding-window causal attention, computed tile by tile over the query
    axis. For each query tile [qs, qe), only the key/value slice
    [max(0, qs - window + 1), qe) is ever touched -- the full (n, n) mask
    or score matrix is never materialized.

    Q, K, V: (n, d) float64 represented as lists of lists.
    window: query i attends to keys max(0, i-window+1) .. i.
    block_size: number of query rows processed per tile.

    Returns: (n, d) float64 attention output as a list of lists.
    """
    n = len(Q)
    d = len(Q[0])
    scale = 1.0 / math.sqrt(d)

    out = [[0.0] * d for _ in range(n)]

    for qs in range(0, n, block_size):
        qe = min(n, qs + block_size)
        k_lo = max(0, qs - window + 1)
        k_hi = qe

        Q_tile = Q[qs:qe]
        K_tile = K[k_lo:k_hi]
        V_tile = V[k_lo:k_hi]

        num_q = qe - qs
        num_k = k_hi - k_lo

        scores = [[0.0] * num_k for _ in range(num_q)]
        for i in range(num_q):
            for j in range(num_k):
                dot = 0.0
                for k_idx in range(d):
                    dot += Q_tile[i][k_idx] * K_tile[j][k_idx]
                scores[i][j] = dot * scale

        masked = [[0.0] * num_k for _ in range(num_q)]
        for i in range(num_q):
            row_idx = qs + i
            for j in range(num_k):
                col_idx = k_lo + j
                allowed = (col_idx <= row_idx) and ((row_idx - col_idx) < window)
                if allowed:
                    masked[i][j] = scores[i][j]
                else:
                    masked[i][j] = -float('inf')

        for i in range(num_q):
            max_val = masked[i][0]
            for j in range(1, num_k):
                if masked[i][j] > max_val:
                    max_val = masked[i][j]
            for j in range(num_k):
                masked[i][j] = masked[i][j] - max_val

        e = [[0.0] * num_k for _ in range(num_q)]
        for i in range(num_q):
            for j in range(num_k):
                e[i][j] = math.exp(masked[i][j])

        p = [[0.0] * num_k for _ in range(num_q)]
        for i in range(num_q):
            sum_e = 0.0
            for j in range(num_k):
                sum_e += e[i][j]
            for j in range(num_k):
                p[i][j] = e[i][j] / sum_e

        res_tile = [[0.0] * d for _ in range(num_q)]
        for i in range(num_q):
            for j in range(d):
                val = 0.0
                for k_idx in range(num_k):
                    val += p[i][k_idx] * V_tile[k_idx][j]
                res_tile[i][j] = val

        for i in range(num_q):
            out[qs + i] = res_tile[i]

    return out
