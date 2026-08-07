import math


def flash_attention_accumulate(
    q: list[float], K: list[list[float]], V: list[list[float]], block_size: int
) -> list[float]:
    d = len(q)
    n = len(K)
    dv = len(V[0]) if n > 0 else 0

    m = -math.inf
    s = 0.0
    acc = [0.0] * dv

    for start in range(0, n, block_size):
        end = min(start + block_size, n)
        scores = []
        for i in range(start, end):
            row = K[i]
            dot = sum(row[k] * q[k] for k in range(d))
            scores.append(dot)

        block_m = max(scores) if scores else -math.inf

        if block_m > m:
            if math.isfinite(m):
                scale = math.exp(m - block_m)
                s *= scale
                for j in range(dv):
                    acc[j] *= scale
            m = block_m

        weights = [math.exp(score - m) for score in scores]
        s += sum(weights)

        for i_rel, weight in enumerate(weights):
            i_abs = start + i_rel
            v_row = V[i_abs]
            for j in range(dv):
                acc[j] += weight * v_row[j]

    return [val / s for val in acc]
