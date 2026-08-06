import numpy as np


def _softmax(x):
    x = x - np.max(x, axis=-1, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=-1, keepdims=True)


def mha_forward(X, Wq, Wk, Wv, Wo, num_heads):
    B, S, E = X.shape
    d = E // num_heads

    q = (X @ Wq).reshape(B, S, num_heads, d).transpose(0, 2, 1, 3)
    k = (X @ Wk).reshape(B, S, num_heads, d).transpose(0, 2, 1, 3)
    v = (X @ Wv).reshape(B, S, num_heads, d).transpose(0, 2, 1, 3)

    scores = q @ np.swapaxes(k, -1, -2) / np.sqrt(d)
    weights = _softmax(scores)
    out = weights @ v

    out = out.transpose(0, 2, 1, 3).reshape(B, S, E)
    return out @ Wo
