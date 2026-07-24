import numpy as np


def flash_backward_dq_dk_dv(Q, K, V, dO, logsumexp, causal):
    n, d = Q.shape
    scale = np.sqrt(d)
    scores = Q @ K.T / scale
    if causal:
        scores = np.where(np.triu(np.ones((n, n), dtype=bool), 1), -np.inf, scores)

    P = np.exp(scores - logsumexp[:, None])

    dV = P.T @ dO
    dP = dO @ V.T
    delta = np.sum(dP * P, axis=1, keepdims=True)
    dS = P * (dP - delta)

    dQ = dS @ K / scale
    dK = dS.T @ Q / scale
    return dQ, dK, dV
