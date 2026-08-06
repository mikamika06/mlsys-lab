import math


def windowed_ring_attention(
    Q: list[list[float]],
    K: list[list[float]],
    V: list[list[float]],
    W: int,
) -> tuple[list[list[float]], list[list[float]], list[list[float]]]:
    """Sliding-window attention backed by a fixed-size ring buffer."""
    n = len(Q)
    d = len(Q[0])
    dv = len(V[0])
    scale = math.sqrt(d)

    Kbuf = [[0.0] * d for _ in range(W)]
    Vbuf = [[0.0] * dv for _ in range(W)]
    out = [[0.0] * dv for _ in range(n)]

    filled = 0
    for t in range(n):
        slot = t % W
        for j in range(d):
            Kbuf[slot][j] = Q_or_K_val = K[t][j]
        for j in range(dv):
            Vbuf[slot][j] = V[t][j]

        if filled + 1 < W:
            filled = filled + 1
        else:
            filled = W

        logits = []
        for i in range(filled):
            dot_val = 0.0
            for j in range(d):
                dot_val += Kbuf[i][j] * Q[t][j]
            logits.append(dot_val / scale)

        max_logit = logits[0]
        for val in logits:
            if val > max_logit:
                max_logit = val

        p = []
        sum_exp = 0.0
        for val in logits:
            exp_val = math.exp(val - max_logit)
            p.append(exp_val)
            sum_exp += exp_val

        for i in range(filled):
            p[i] /= sum_exp

        for j in range(dv):
            out_j = 0.0
            for i in range(filled):
                out_j += p[i] * Vbuf[i][j]
            out[t][j] = out_j

    return out, Kbuf, Vbuf
