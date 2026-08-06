import math
import numpy as np


def windowed_ring_attention(Q, K, V, W):
    """Sliding-window attention backed by a fixed-size ring buffer."""
    Q = np.asarray(Q, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)

    n, d = Q.shape
    dv = V.shape[1]
    scale = math.sqrt(d)

    Kbuf = np.zeros((W, d), dtype=np.float64)
    Vbuf = np.zeros((W, dv), dtype=np.float64)
    out = np.empty((n, dv), dtype=np.float64)

    filled = 0
    for t in range(n):
        slot = t % W
        for j in range(d):
            Kbuf[slot, j] = K[t, j]
        for j in range(dv):
            Vbuf[slot, j] = V[t, j]

        if filled + 1 < W:
            filled = filled + 1
        else:
            filled = W

        logits = []
        for i in range(filled):
            dot_val = 0.0
            for j in range(d):
                dot_val += Kbuf[i, j] * Q[t, j]
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
                out_j += p[i] * Vbuf[i, j]
            out[t, j] = out_j

    return out, Kbuf, Vbuf
