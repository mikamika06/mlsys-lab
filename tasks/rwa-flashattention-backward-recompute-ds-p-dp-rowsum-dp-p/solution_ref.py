import numpy as np


def flash_attention_backward(q, k, v, do, m, l):
    q = np.asarray(q, dtype=np.float64)
    k = np.asarray(k, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    do = np.asarray(do, dtype=np.float64)
    m = np.asarray(m, dtype=np.float64)
    l = np.asarray(l, dtype=np.float64)

    scores = q @ k.T
    p = np.exp(scores - m) / l

    dp = do @ v.T
    rowsum = np.sum(dp * p, axis=1, keepdims=True)
    ds = p * (dp - rowsum)

    dq = ds @ k
    dk = ds.T @ q
    dv = p.T @ do

    return dq, dk, dv
