import numpy as np


def gptq_quantize_weight(W, Hinv, quantize_fn):
    W = W.copy()
    Hinv = Hinv.copy()
    rows, cols = W.shape

    for i in range(cols):
        w = W[:, i].copy()
        d = Hinv[i, i]
        q = quantize_fn(w)

        err = (w - q) / d
        W[:, i] = q

        if i + 1 < cols:
            W[:, i+1:] -= np.outer(err, Hinv[i, i+1:])
            Hinv[i+1:, i+1:] -= np.outer(Hinv[i+1:, i], Hinv[i, i+1:]) / d

    return W
