import numpy as np


def flash_backward(Q, K, V, dO, m, l):
    Q = np.asarray(Q, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    dO = np.asarray(dO, dtype=np.float64)
    m = np.asarray(m, dtype=np.float64)
    l = np.asarray(l, dtype=np.float64)

    scale = np.sqrt(float(Q.shape[1]))
    S = Q @ K.T / scale
    P = np.exp(S - m[:, None]) / l[:, None]

    dP = dO @ V.T
    correction = np.sum(dP * P, axis=1, keepdims=True)
    dS = P * (dP - correction)

    dQ = (dS @ K) / scale
    dK = (dS.T @ Q) / scale
    dV = P.T @ dO

    return (
        np.asarray(dQ, dtype=np.float64),
        np.asarray(dK, dtype=np.float64),
        np.asarray(dV, dtype=np.float64),
    )
