import numpy as np


def attention_checkpoint(Q, K, V, G):
    d = Q.shape[1]
    scale = np.sqrt(d)

    scores = (Q @ K.T) / scale
    scores = scores - np.max(scores, axis=1, keepdims=True)
    exp_scores = np.exp(scores)
    P = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

    dV = P.T @ G
    dP = G @ V.T
    ds = P * (dP - np.sum(dP * P, axis=1, keepdims=True))

    dQ = (ds @ K) / scale
    dK = (ds.T @ Q) / scale

    reported_memory = int(Q.nbytes + K.nbytes + V.nbytes)
    return dQ.astype(np.float64), dK.astype(np.float64), dV.astype(np.float64), reported_memory
