import math
import numpy as np


def _softmax_rows(x: np.ndarray) -> np.ndarray:
    rows, cols = x.shape
    out = np.zeros((rows, cols), dtype=np.float64)
    for i in range(rows):
        m = x[i, 0]
        for j in range(1, cols):
            if x[i, j] > m:
                m = x[i, j]
        s = 0.0
        exps = [0.0] * cols
        for j in range(cols):
            v = math.exp(x[i, j] - m)
            exps[j] = v
            s += v
        for j in range(cols):
            out[i, j] = exps[j] / s
    return out


def _attend(q: np.ndarray, K: np.ndarray, V: np.ndarray, d: int) -> np.ndarray:
    n_keys = K.shape[0]
    scores = np.zeros((1, n_keys), dtype=np.float64)
    sqrt_d = math.sqrt(d)
    for j in range(n_keys):
        s = 0.0
        for k in range(d):
            s += q[k] * K[j, k]
        scores[0, j] = s / sqrt_d
    weights = _softmax_rows(scores)[0]
    out = np.zeros(d, dtype=np.float64)
    for k in range(d):
        s = 0.0
        for j in range(n_keys):
            s += weights[j] * V[j, k]
        out[k] = s
    return out


def snapkv_pooled_selection(K: np.ndarray, V: np.ndarray, Q_obs: np.ndarray, Q_new: np.ndarray,
                             budget: int, pool_size: int) -> dict:
    """SnapKV KV-cache compression, applied independently per attention
    head (each head may keep a different subset of positions).
    """
    K = np.asarray(K, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    Q_obs = np.asarray(Q_obs, dtype=np.float64)
    Q_new = np.asarray(Q_new, dtype=np.float64)

    H, n, d = K.shape
    w = Q_obs.shape[1]
    pad = pool_size // 2

    kept_idx = []
    outputs = np.zeros((H, d), dtype=np.float64)
    sqrt_d = math.sqrt(d)

    for h in range(H):
        mat = np.zeros((w, n), dtype=np.float64)
        for i in range(w):
            for j in range(n):
                s = 0.0
                for k in range(d):
                    s += Q_obs[h, i, k] * K[h, j, k]
                mat[i, j] = s / sqrt_d

        attn = _softmax_rows(mat)

        raw_score = [0.0] * n
        for j in range(n):
            s = 0.0
            for i in range(w):
                s += attn[i, j]
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

        idx = np.array(idx_list)
        kept_idx.append(idx)
        outputs[h] = _attend(Q_new[h], K[h][idx], V[h][idx], d)

    return {"kept_idx": kept_idx, "output": outputs}
