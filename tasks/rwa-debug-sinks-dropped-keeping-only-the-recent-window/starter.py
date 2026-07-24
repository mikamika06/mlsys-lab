import numpy as np

def streaming_attention(q, k, v, window_size=4):
    T, d = q.shape
    out = []
    for t in range(T):
        q_t = q[t:t+1]
        # Buggy: simple sliding window that evicts the oldest token when full
        start = max(0, t - window_size + 1)
        end = t + 1
        K = k[start:end]
        V = v[start:end]
        scores = q_t @ K.T / np.sqrt(d)
        scores -= scores.max(axis=1, keepdims=True)
        weights = np.exp(scores)
        weights = weights / weights.sum(axis=1, keepdims=True)
        out.append(weights @ V)
    return np.vstack(out)
