import math
import numpy as np


def _quantize(x, kind):
    h, t, d = x.shape
    if kind == 0:
        max_val = 0.0
        for i in range(h):
            for j in range(t):
                for k in range(d):
                    v = abs(float(x[i, j, k]))
                    if v > max_val:
                        max_val = v
        scale = max_val / 127.0
        out_data = [[[0.0] * d for _ in range(t)] for _ in range(h)]
        for i in range(h):
            for j in range(t):
                for k in range(d):
                    val = 0.0 if scale == 0.0 else float(x[i, j, k]) / scale
                    rounded = round(val)
                    if rounded < -127:
                        rounded = -127
                    elif rounded > 127:
                        rounded = 127
                    out_data[i][j][k] = rounded * scale
        return np.array(out_data, dtype=x.dtype), np.array([scale], dtype=x.dtype)
    if kind == 1:
        scale_data = [[[0.0] for _ in range(t)] for _ in range(1)]
        for j in range(t):
            max_val = 0.0
            for i in range(h):
                for k in range(d):
                    v = abs(float(x[i, j, k]))
                    if v > max_val:
                        max_val = v
            scale_data[0][j][0] = max_val / 127.0
        out_data = [[[0.0] * d for _ in range(t)] for _ in range(h)]
        for i in range(h):
            for j in range(t):
                s = scale_data[0][j][0]
                for k in range(d):
                    val = 0.0 if s == 0.0 else float(x[i, j, k]) / s
                    rounded = round(val)
                    if rounded < -127:
                        rounded = -127
                    elif rounded > 127:
                        rounded = 127
                    out_data[i][j][k] = rounded * s
        flat_scale = np.array([scale_data[0][j][0] for j in range(t)], dtype=x.dtype)
        return np.array(out_data, dtype=x.dtype), flat_scale
    scale_data = [[[0.0] for _ in range(1)] for _ in range(h)]
    for i in range(h):
        max_val = 0.0
        for j in range(t):
            for k in range(d):
                v = abs(float(x[i, j, k]))
                if v > max_val:
                    max_val = v
        scale_data[i][0][0] = max_val / 127.0
    out_data = [[[0.0] * d for _ in range(t)] for _ in range(h)]
    for i in range(h):
        s = scale_data[i][0][0]
        for j in range(t):
            for k in range(d):
                val = 0.0 if s == 0.0 else float(x[i, j, k]) / s
                rounded = round(val)
                if rounded < -127:
                    rounded = -127
                elif rounded > 127:
                    rounded = 127
                out_data[i][j][k] = rounded * s
    flat_scale = np.array([scale_data[i][0][0] for i in range(h)], dtype=x.dtype)
    return np.array(out_data, dtype=x.dtype), flat_scale


def _attention(Q, K, V):
    h, t_q, d = Q.shape
    _, t_k, _ = K.shape
    scores_data = [[[0.0] * t_k for _ in range(t_q)] for _ in range(h)]
    sqrt_d = math.sqrt(d)
    for i in range(h):
        for r in range(t_q):
            for c in range(t_k):
                s = 0.0
                for k in range(d):
                    s += float(Q[i, r, k]) * float(K[i, c, k])
                scores_data[i][r][c] = s / sqrt_d
    for i in range(h):
        for r in range(t_q):
            max_score = scores_data[i][r][0]
            for c in range(1, t_k):
                if scores_data[i][r][c] > max_score:
                    max_score = scores_data[i][r][c]
            for c in range(t_k):
                scores_data[i][r][c] = math.exp(scores_data[i][r][c] - max_score)
    for i in range(h):
        for r in range(t_q):
            total = 0.0
            for c in range(t_k):
                total += scores_data[i][r][c]
            if total != 0.0:
                for c in range(t_k):
                    scores_data[i][r][c] /= total
    out_v = [[[0.0] * d for _ in range(t_q)] for _ in range(h)]
    for i in range(h):
        for r in range(t_q):
            for col in range(d):
                val = 0.0
                for c in range(t_k):
                    val += scores_data[i][r][c] * float(V[i, c, col])
                out_v[i][r][col] = val
    return np.array(out_v, dtype=Q.dtype)


def choose_kv_scale_granularity(K, V, Q, budget):
    ref = _attention(Q, K, V)
    best_cost = None
    best_idx = 0
    for i in range(3):
        Kq, ks = _quantize(K, i)
        Vq, vs = _quantize(V, i)
        out = _attention(Q, Kq, Vq)
        h, t_q, d = ref.shape
        mse_sum = 0.0
        total_elements = h * t_q * d
        for bh in range(h):
            for r in range(t_q):
                for col in range(d):
                    diff = float(ref[bh, r, col]) - float(out[bh, r, col])
                    mse_sum += diff * diff
        mse = mse_sum / total_elements
        scale_bytes = (ks.size + vs.size) * 4
        cost = float(mse + 0.001 * max(0, scale_bytes - budget))
        if best_cost is None or cost < best_cost:
            best_cost = cost
            best_idx = i
    return best_idx
