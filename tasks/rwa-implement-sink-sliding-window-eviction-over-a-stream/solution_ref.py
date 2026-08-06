import math
import numpy as np


def sink_attention_stream(Q, K, V, k, w):
    Q = np.asarray(Q, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)

    n, d = Q.shape
    m = V.shape[1]
    outputs = []
    lengths = []

    cache = []
    for t in range(n):
        cache.append(t)

        min_recent = max(0, t - w + 1)
        cache = [
            idx for idx in cache
            if idx < k or idx >= min_recent
        ]

        lengths.append(len(cache))

        scale = math.sqrt(d)
        logits = []
        for idx in cache:
            dot_val = 0.0
            q_t = Q[t]
            k_idx = K[idx]
            for a, b in zip(k_idx, q_t):
                dot_val += a * b
            logits.append(dot_val / scale)

        max_logit = logits[0]
        for val in logits:
            if val > max_logit:
                max_logit = val

        logits_shifted = [val - max_logit for val in logits]
        probs = []
        sum_probs = 0.0
        for val in logits_shifted:
            e = math.exp(val)
            probs.append(e)
            sum_probs += e

        probs = [p / sum_probs for p in probs]

        out_vec = [0.0] * m
        for i, p in enumerate(probs):
            v_row = V[cache[i]]
            for j in range(m):
                out_vec[j] += p * v_row[j]
        outputs.append(out_vec)

    return np.asarray(outputs, dtype=np.float64), lengths
