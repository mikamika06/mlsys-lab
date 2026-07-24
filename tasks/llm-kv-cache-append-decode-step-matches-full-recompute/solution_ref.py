import numpy as np


def _softmax(x):
    x = np.asarray(x, dtype=np.float64)
    x = x - np.max(x, axis=-1, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=-1, keepdims=True)


def decode_steps(x, Wq, Wk, Wv):
    x = np.asarray(x, dtype=np.float64)
    Wq = np.asarray(Wq, dtype=np.float64)
    Wk = np.asarray(Wk, dtype=np.float64)
    Wv = np.asarray(Wv, dtype=np.float64)

    k_cache = []
    v_cache = []
    outputs = []
    scale = np.sqrt(Wq.shape[1])

    for token in x:
        q = token @ Wq
        k = token @ Wk
        v = token @ Wv

        k_cache.append(k)
        v_cache.append(v)

        keys = np.stack(k_cache, axis=0)
        values = np.stack(v_cache, axis=0)

        weights = _softmax((q[None, :] @ keys.T) / scale)
        outputs.append(weights @ values)

    return np.concatenate(outputs, axis=0)
