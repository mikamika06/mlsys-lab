import numpy as np

def streaming_attention(q, k, v, window_size=4):
    T, d = q.shape
    out = []
    for t in range(T):
        q_t = q[t:t+1]
        if t == 0:
            indices = [0]
        else:
            start = max(1, t - window_size + 2)
            indices = [0] + list(range(start, t + 1))
        K = k[indices]
        V = v[indices]
        scores = q_t @ K.T / np.sqrt(d)
        scores -= scores.max(axis=1, keepdims=True)
        weights = np.exp(scores)
        weights = weights / weights.sum(axis=1, keepdims=True)
        out.append(weights @ V)
    return np.vstack(out)
