import math


def _rope(x: list[list[float]], positions: list[float], theta: float) -> list[list[float]]:
    n = len(x)
    d = len(x[0]) if n > 0 else 0
    half = d // 2
    freq = [0.0] * half
    for i in range(half):
        freq[i] = theta ** (-i * 2.0 / d)
    out_list = [[0.0] * d for _ in range(n)]
    for i in range(n):
        pos = float(positions[i])
        for j in range(d):
            out_list[i][j] = float(x[i][j])
        for h in range(half):
            angle = pos * freq[h]
            c = math.cos(angle)
            s = math.sin(angle)
            idx0 = 2 * h
            idx1 = 2 * h + 1
            x0 = float(x[i][idx0])
            x1 = float(x[i][idx1])
            out_list[i][idx0] = x0 * c - x1 * s
            out_list[i][idx1] = x0 * s + x1 * c
    return out_list


def streaming_rope_attention(
    q: list[list[float]],
    k: list[list[float]],
    v: list[list[float]],
    kept_indices: list[int],
    theta: float = 10000.0,
) -> list[list[float]]:
    q_kept = [q[idx] for idx in kept_indices]
    k_kept = [k[idx] for idx in kept_indices]
    v_kept = [v[idx] for idx in kept_indices]
    positions = [float(i) for i in range(len(kept_indices))]

    qr = _rope(q_kept, positions, theta)
    kr = _rope(k_kept, positions, theta)

    n_q = len(qr)
    n_k = len(kr)
    d = len(qr[0]) if n_q > 0 else 0
    scale = math.sqrt(d)

    scores = [[0.0] * n_k for _ in range(n_q)]
    for i in range(n_q):
        for j in range(n_k):
            dot = 0.0
            for k_idx in range(d):
                dot += float(qr[i][k_idx]) * float(kr[j][k_idx])
            scores[i][j] = dot / scale

    for i in range(n_q):
        max_val = scores[i][0]
        for j in range(1, n_k):
            if scores[i][j] > max_val:
                max_val = scores[i][j]
        for j in range(n_k):
            scores[i][j] -= max_val

    weights = [[0.0] * n_k for _ in range(n_q)]
    for i in range(n_q):
        for j in range(n_k):
            weights[i][j] = math.exp(scores[i][j])

    for i in range(n_q):
        total = 0.0
        for j in range(n_k):
            total += weights[i][j]
        for j in range(n_k):
            weights[i][j] /= total

    v_rows = len(v_kept)
    v_cols = len(v_kept[0]) if v_rows > 0 else 0
    out_v = [[0.0] * v_cols for _ in range(n_q)]
    for i in range(n_q):
        for j in range(v_cols):
            val = 0.0
            for k_idx in range(v_rows):
                val += weights[i][k_idx] * float(v_kept[k_idx][j])
            out_v[i][j] = val

    return out_v
