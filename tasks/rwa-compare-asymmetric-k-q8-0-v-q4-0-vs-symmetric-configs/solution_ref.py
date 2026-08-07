from __future__ import annotations

import math


def _quantize_symmetric(x: list[list[float]], bits: int) -> list[list[float]]:
    qmax = (1 << (bits - 1)) - 1

    max_val = 0.0
    for row in x:
        for val in row:
            abs_val = val if val >= 0.0 else -val
            if abs_val > max_val:
                max_val = abs_val

    scale = max_val / qmax
    if scale == 0.0:
        out = []
        for row in x:
            out.append([0.0 for _ in row])
        return out

    out = []
    for row in x:
        new_row = []
        for val in row:
            new_row.append(round(val / scale) * scale)
        out.append(new_row)
    return out


def _attention(K: list[list[float]], V: list[list[float]], q: list[float]) -> list[float]:
    n_rows = len(K)
    n_cols = len(K[0]) if n_rows > 0 else 0

    logits = []
    for i in range(n_rows):
        s = 0.0
        for j in range(n_cols):
            s += K[i][j] * q[j]
        logits.append(s / math.sqrt(n_cols))

    max_logit = logits[0]
    for val in logits[1:]:
        if val > max_logit:
            max_logit = val

    p = []
    sum_p = 0.0
    for l in logits:
        val = math.exp(l - max_logit)
        p.append(val)
        sum_p += val

    p = [val / sum_p for val in p]

    v_cols = len(V[0]) if len(V) > 0 else 0
    result = [0.0] * v_cols
    for j in range(v_cols):
        s = 0.0
        for i in range(n_rows):
            s += p[i] * V[i][j]
        result[j] = s

    return result


def kv_config_attention_errors(K: list[list[float]], V: list[list[float]], q: list[float]) -> list[float]:
    base = _attention(K, V, q)
    result = []
    for kb, vb in [(8, 8), (4, 4), (8, 4)]:
        kq = _quantize_symmetric(K, kb)
        vq = _quantize_symmetric(V, vb)
        att = _attention(kq, vq, q)
        max_diff = 0.0
        for a_val, b_val in zip(att, base):
            diff = a_val - b_val
            abs_diff = diff if diff >= 0.0 else -diff
            if abs_diff > max_diff:
                max_diff = abs_diff
        result.append(float(max_diff))
    return result
