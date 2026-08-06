import math
import numpy as np

def streaming_attention(q, k, v, window_size=4):
    T, d = q.shape
    out = []
    sqrt_d = math.sqrt(d)
    for t in range(T):
        q_t = q[t:t+1]
        if t == 0:
            indices = [0]
        else:
            start = max(1, t - window_size + 2)
            indices = [0] + list(range(start, t + 1))
        K = k[indices]
        V = v[indices]
        n_indices = len(indices)
        scores_row = [0.0] * n_indices
        for j in range(n_indices):
            dot_val = 0.0
            for m in range(d):
                dot_val += q_t[0, m] * K[j, m]
            scores_row[j] = dot_val / sqrt_d
        max_score = scores_row[0]
        for val in scores_row:
            if val > max_score:
                max_score = val
        for j in range(n_indices):
            scores_row[j] -= max_score
        weights_row = [0.0] * n_indices
        for j in range(n_indices):
            weights_row[j] = math.exp(scores_row[j])
        sum_weights = 0.0
        for val in weights_row:
            sum_weights += val
        for j in range(n_indices):
            weights_row[j] /= sum_weights
        row_out = np.zeros((1, d), dtype=q.dtype)
        for c in range(d):
            acc = 0.0
            for j in range(n_indices):
                acc += weights_row[j] * V[j, c]
            row_out[0, c] = acc
        out.append(row_out)
    return np.vstack(out)
