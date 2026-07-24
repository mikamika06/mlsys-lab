import numpy as np


def _softmax(x):
    x = x - np.max(x, axis=-1, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=-1, keepdims=True)


def mha_forward(X, Wq, Wk, Wv, Wo, num_heads):
    B, S, E = X.shape
    d = E // num_heads

    # TODO: this reshape keeps the wrong layout. It treats the sequence axis
    # as if it were already the head axis and skips the required transpose.
    q = (X @ Wq).reshape(B, num_heads, S, d)
    k = (X @ Wk).reshape(B, num_heads, S, d)
    v = (X @ Wv).reshape(B, num_heads, S, d)

    scores = q @ np.swapaxes(k, -1, -2) / np.sqrt(d)
    weights = _softmax(scores)
    out = weights @ v

    out = out.reshape(B, S, E)
    return out @ Wo
