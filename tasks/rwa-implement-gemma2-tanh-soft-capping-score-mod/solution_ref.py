import numpy as np


def attention_with_score_mod(Q, K, V, cap):
    Q = np.asarray(Q, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)

    scores = (Q @ K.T) / np.sqrt(Q.shape[1])
    scores = cap * np.tanh(scores / cap)

    scores = scores - np.max(scores, axis=1, keepdims=True)
    weights = np.exp(scores)
    weights = weights / np.sum(weights, axis=1, keepdims=True)

    return weights @ V
