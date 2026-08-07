import math


def _softmax_rows(x: list[list[float]]) -> list[list[float]]:
    rows = len(x)
    cols = len(x[0])
    out = [[0.0] * cols for _ in range(rows)]
    for i in range(rows):
        m = x[i][0]
        for j in range(1, cols):
            if x[i][j] > m:
                m = x[i][j]
        s = 0.0
        exps = [0.0] * cols
        for j in range(cols):
            v = math.exp(x[i][j] - m)
            exps[j] = v
            s += v
        for j in range(cols):
            out[i][j] = exps[j] / s
    return out


def _attend(q: list[float], K: list[list[float]], V: list[list[float]], d: int) -> list[float]:
    n_keys = len(K)
    scores = [0.0] * n_keys
    sqrt_d = math.sqrt(d)
    for j in range(n_keys):
        s = 0.0
        for k in range(d):
            s += q[k] * K[j][k]
        scores[j] = s / sqrt_d
    weights = _softmax_rows([scores])[0]
    out = [0.0] * d
    for k in range(d):
        s = 0.0
        for j in range(n_keys):
            s += weights[j] * V[j][k]
        out[k] = s
    return out


def snapkv_pooled_selection(K: list[list[list[float]]], V: list[list[list[float]]],
                             Q_obs: list[list[list[float]]], Q_new: list[list[float]],
                             budget: int, pool_size: int) -> dict:
    """SnapKV KV-cache compression, applied independently per attention
    head (each head may keep a different subset of positions).
    """
    H = len(K)
    n = len(K[0])
    d = len(K[0][0])
    w = len(Q_obs[0])
    pad = pool_size // 2

    kept_idx = []
    outputs = [[0.0] * d for _ in range(H)]
    sqrt_d = math.sqrt(d)

    for h in range(H):
        mat = [[0.0] * n for _ in range(w)]
        for i in range(w):
            for j in range(n):
                s = 0.0
                for k in range(d):
                    s += Q_obs[h][i][k] * K[h][j][k]
                mat[i][j] = s / sqrt_d

        attn = _softmax_rows(mat)

        raw_score = [0.0] * n
        for j in range(n):
            s = 0.0
            for i in range(w):
                s += attn[i][j]
            raw_score[j] = s

        padded = [0.0] * (n + 2 * pad)
        for i in range(pad):
            padded[i] = raw_score[0]
        for i in range(n):
            padded[pad + i] = raw_score[i]
        for i in range(pad):
            padded[pad + n + i] = raw_score[n - 1]

        pooled = [0.0] * n
        inv_pool = 1.0 / pool_size
        for j in range(n):
            s = 0.0
            for m in range(pool_size):
                s += padded[j + m] * inv_pool
            pooled[j] = s

        win = list(range(n - w, n))
        k_extra = budget - w
        if k_extra <= 0:
            idx_list = sorted(win[-budget:])
        else:
            cand = list(range(n - w))
            top_extra = sorted(cand, key=lambda i: pooled[i], reverse=True)[:k_extra]
            idx_list = sorted(win + top_extra)

        kept_idx.append(idx_list)

        sub_K = [K[h][idx] for idx in idx_list]
        sub_V = [V[h][idx] for idx in idx_list]
        outputs[h] = _attend(Q_new[h], sub_K, sub_V, d)

    return {"kept_idx": kept_idx, "output": outputs}
