import math


def chunked_attention(q: list[float], chunks: list[tuple[list[list[float]], list[list[float]]]]) -> list[float]:
    m = -math.inf
    l = 0.0
    h = len(chunks[0][1][0])
    out = [0.0] * h

    for K, V in chunks:
        scores = []
        for row in K:
            s = sum(a * b for a, b in zip(row, q))
            scores.append(s)

        chunk_max = max(scores)
        new_m = max(m, chunk_max)

        old_scale = 0.0 if m == -math.inf else math.exp(m - new_m)
        weights = [math.exp(s - new_m) for s in scores]

        new_l_old_scale = l * old_scale
        new_out = [val * new_l_old_scale for val in out]
        for w, v_row in zip(weights, V):
            for j in range(h):
                new_out[j] += w * v_row[j]
        out = new_out

        sum_weights = sum(weights)
        l = new_l_old_scale + sum_weights
        m = new_m

        out = [val / l for val in out]

    return out
