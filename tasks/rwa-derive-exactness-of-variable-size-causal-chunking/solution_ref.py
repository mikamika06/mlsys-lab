import math
import numpy as np


def causal_chunk_attention(Q, K, V, chunks):
    Q = np.asarray(Q, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)

    N, D = Q.shape
    M = V.shape[1]
    out = np.empty((N, M), dtype=np.float64)
    scale = math.sqrt(D)

    start = 0
    for size in chunks:
        end = start + size
        for i in range(size):
            q_row_idx = start + i
            scores = []
            for j in range(end):
                if j > q_row_idx:
                    scores.append(-float("inf"))
                else:
                    dot = 0.0
                    for d in range(D):
                        dot += Q[q_row_idx, d] * K[j, d]
                    scores.append(dot / scale)

            max_score = -float("inf")
            for s in scores:
                if s > max_score:
                    max_score = s

            weights = []
            weight_sum = 0.0
            for s in scores:
                w = math.exp(s - max_score)
                weights.append(w)
                weight_sum += w

            norm_weights = []
            for w in weights:
                norm_weights.append(w / weight_sum)

            for m in range(M):
                val = 0.0
                for j in range(end):
                    val += norm_weights[j] * V[j, m]
                out[q_row_idx, m] = val

        start = end

    return out
