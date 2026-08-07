import math


def _softmax(x: list[list[float]]) -> list[list[float]]:
    n_rows = len(x)
    n_cols = len(x[0]) if n_rows > 0 else 0
    out = [[0.0] * n_cols for _ in range(n_rows)]
    for i in range(n_rows):
        m = x[i][0]
        for j in range(1, n_cols):
            if x[i][j] > m:
                m = x[i][j]
        s = 0.0
        for j in range(n_cols):
            val = math.exp(x[i][j] - m)
            out[i][j] = val
            s += val
        for j in range(n_cols):
            out[i][j] /= s
    return out


def _attention(Q: list[list[float]], K: list[list[float]], V: list[list[float]]) -> list[list[float]]:
    n_q = len(Q)
    d = len(Q[0]) if n_q > 0 else 0
    n_k = len(K)
    d_v = len(V[0]) if len(V) > 0 else 0
    sqrt_d = math.sqrt(d)
    scores = [[0.0] * n_k for _ in range(n_q)]
    for i in range(n_q):
        for j in range(n_k):
            acc = 0.0
            for l in range(d):
                acc += Q[i][l] * K[j][l]
            scores[i][j] = acc / sqrt_d
    sm = _softmax(scores)
    out = [[0.0] * d_v for _ in range(n_q)]
    for i in range(n_q):
        for j in range(d_v):
            acc = 0.0
            for l in range(n_k):
                acc += sm[i][l] * V[l][j]
            out[i][j] = acc
    return out


def optimize_sink_window_split(Q: list[list[float]], K: list[list[float]], V: list[list[float]], B: int) -> int:
    full = _attention(Q, K, V)
    n = len(Q)

    best_k = 1
    best_error = float("inf")

    for k in range(1, B):
        w = B - k
        idx_list = []
        for i in range(k):
            idx_list.append(i)
        for i in range(n - w, n):
            idx_list.append(i)

        seen = set()
        indices = []
        for idx in idx_list:
            if idx not in seen:
                seen.add(idx)
                indices.append(idx)
        indices.sort()

        sub_K = [K[i] for i in indices]
        sub_V = [V[i] for i in indices]

        approx = _attention(Q, sub_K, sub_V)

        error = 0.0
        n_rows = len(full)
        n_cols = len(full[0]) if n_rows > 0 else 0
        for i in range(n_rows):
            for j in range(n_cols):
                diff = full[i][j] - approx[i][j]
                error += diff * diff

        if error < best_error:
            best_error = error
            best_k = k

    return int(best_k)
