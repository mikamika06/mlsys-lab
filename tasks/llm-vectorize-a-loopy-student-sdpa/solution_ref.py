import numpy as np


def sdpa(Q, K, V):
    d = Q.shape[1]
    scores = Q @ K.T / np.sqrt(d)
    scores = scores - np.max(scores, axis=1, keepdims=True)
    probs = np.exp(scores)
    probs = probs / np.sum(probs, axis=1, keepdims=True)
    return probs @ V
