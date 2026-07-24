import numpy as np


def flash_forward_reconstruct(Q, K, V, m, l):
    Q = np.asarray(Q, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    m = np.asarray(m, dtype=np.float64)
    l = np.asarray(l, dtype=np.float64)

    d = Q.shape[-1]
    S = (Q @ K.T) / np.sqrt(d)

    # Reuse the supplied (m, l) directly -- no second max pass.
    P = np.exp(S - m[:, None]) / l[:, None]
    O = P @ V

    return P, O
