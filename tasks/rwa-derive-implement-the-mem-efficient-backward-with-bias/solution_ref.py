import numpy as np


def biased_flash_backward(Q, K, V, B, dO, m, l):
    """Memory-efficient attention backward with an additive bias (e.g.
    ALiBi or a mask bias), given only the saved row statistics (m, l)
    from the forward pass -- never a cached probability matrix.

    Q, K, V   : (n, d)
    B         : (n, n) additive bias, added to the scaled scores before
                softmax on the forward pass. Fixed (no gradient wanted).
    dO        : (n, d) upstream gradient w.r.t. the forward output O.
    m, l      : (n,) row max and row softmax-normalizer saved during the
                forward pass, i.e. S = Q@K.T/sqrt(d) + B,
                m = rowmax(S), l = rowsum(exp(S - m)).

    Returns (dQ, dK, dV), each shaped like Q, K, V respectively.
    """
    Q = np.asarray(Q, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    dO = np.asarray(dO, dtype=np.float64)
    m = np.asarray(m, dtype=np.float64)
    l = np.asarray(l, dtype=np.float64)

    scale = np.sqrt(float(Q.shape[1]))

    # Recompute P from Q, K, B and the saved (m, l) -- the full P matrix
    # was never stored after the forward pass.
    S = Q @ K.T / scale + B
    P = np.exp(S - m[:, None]) / l[:, None]

    dV = P.T @ dO
    dP = dO @ V.T
    correction = np.sum(dP * P, axis=1, keepdims=True)
    dS = P * (dP - correction)  # bias is additive, so dS == d(scaled scores)

    dQ = (dS @ K) / scale
    dK = (dS.T @ Q) / scale

    return (
        np.asarray(dQ, dtype=np.float64),
        np.asarray(dK, dtype=np.float64),
        np.asarray(dV, dtype=np.float64),
    )
