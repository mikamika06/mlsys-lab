import math


def _rope(x: list[list[float]], positions: list[int]) -> list[list[float]]:
    N = len(x)
    if N == 0:
        return []
    d = len(x[0])
    half = d // 2
    freqs = [1.0 / (10000.0 ** (2 * j / d)) for j in range(half)]

    out_x = [[val for val in row] for row in x]
    for i in range(N):
        pos = positions[i]
        for j in range(half):
            angle = pos * freqs[j]
            c = math.cos(angle)
            s = math.sin(angle)
            even_val = out_x[i][2 * j]
            odd_val = out_x[i][2 * j + 1]
            out_x[i][2 * j] = even_val * c - odd_val * s
            out_x[i][2 * j + 1] = even_val * s + odd_val * c
    return out_x


def _attention(q: list[list[float]], k: list[list[float]], v: list[list[float]]) -> list[list[float]]:
    N = len(q)
    if N == 0:
        return []
    d = len(q[0])
    scale = 1.0 / math.sqrt(d)

    logits = [[0.0] * N for _ in range(N)]
    for i in range(N):
        for j in range(N):
            dot = 0.0
            for k_idx in range(d):
                dot += q[i][k_idx] * k[j][k_idx]
            logits[i][j] = dot * scale

    weights = [[0.0] * N for _ in range(N)]
    for i in range(N):
        m = logits[i][0]
        for j in range(1, N):
            if logits[i][j] > m:
                m = logits[i][j]
        row_sum = 0.0
        for j in range(N):
            val = math.exp(logits[i][j] - m)
            weights[i][j] = val
            row_sum += val
        for j in range(N):
            weights[i][j] /= row_sum

    v_d = len(v[0]) if len(v) > 0 else 0
    out = [[0.0] * v_d for _ in range(N)]
    for i in range(N):
        for j in range(v_d):
            val = 0.0
            for k_idx in range(N):
                val += weights[i][k_idx] * v[k_idx][j]
            out[i][j] = val
    return out


def packed_rope_attention(
    q: list[list[float]],
    k: list[list[float]],
    v: list[list[float]],
    cu_seqlens: list[int]
) -> list[list[float]]:
    N = len(q)
    if N == 0:
        return []
    d = len(q[0])
    out = [[0.0] * d for _ in range(N)]

    for i in range(len(cu_seqlens) - 1):
        start = int(cu_seqlens[i])
        end = int(cu_seqlens[i + 1])
        if end == start:
            continue
        positions = list(range(end - start))
        sub_q = [q[idx] for idx in range(start, end)]
        sub_k = [k[idx] for idx in range(start, end)]
        sub_v = [v[idx] for idx in range(start, end)]

        rq = _rope(sub_q, positions)
        rk = _rope(sub_k, positions)
        attn_out = _attention(rq, rk, sub_v)

        for r_idx in range(end - start):
            out[start + r_idx] = attn_out[r_idx]

    return out
