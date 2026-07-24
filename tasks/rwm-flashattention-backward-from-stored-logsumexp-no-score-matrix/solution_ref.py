import numpy as np


def flash_backward(Q, K, V, dO, lse):
    d = Q.shape[1]
    scale = 1.0 / np.sqrt(d)

    scores = (Q @ K.T) * scale
    P = np.exp(scores - lse[:, None])

    dV = P.T @ dO

    dP = dO @ V.T
    correction = np.sum(dP * P, axis=1, keepdims=True)
    dS = P * (dP - correction)

    dQ = (dS @ K) * scale
    dK = (dS.T @ Q) * scale

    return dQ, dK, dV
