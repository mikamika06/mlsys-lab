import math

def streaming_attention(q: list[list[float]], k: list[list[float]], v: list[list[float]], window_size: int = 4) -> list[list[float]]:
    T = len(q)
    d = len(q[0])
    out = []
    for t in range(T):
        q_t = q[t]
        if t == 0:
            indices = [0]
        else:
            start = max(1, t - window_size + 2)
            indices = [0] + list(range(start, t + 1))
        K = [k[i] for i in indices]
        V = [v[i] for i in indices]

        scores = []
        scale = math.sqrt(d)
        for row in K:
            dot = sum(a * b for a, b in zip(q_t, row))
            scores.append(dot / scale)

        max_score = max(scores)
        scores = [s - max_score for s in scores]

        weights = [math.exp(s) for s in scores]
        sum_w = sum(weights)
        weights = [w / sum_w for w in weights]

        res_row = [0.0] * d
        for w, v_row in zip(weights, V):
            for j in range(d):
                res_row[j] += w * v_row[j]
        out.append(res_row)
    return out
