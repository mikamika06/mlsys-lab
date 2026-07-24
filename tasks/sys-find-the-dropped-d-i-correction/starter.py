import numpy as np


def flash_attention_backward(Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                              O: np.ndarray, L: np.ndarray, dO: np.ndarray,
                              scale: float):
    """
    Recompute-based FlashAttention backward: recompute P from Q, K, L
    (never read a stored full attention matrix), then apply the softmax
    VJP to get dQ, dK, dV.

    BUG: this drops the D_i = rowsum(dO * O) correction term when forming
    dS, so dQ and dK come out wrong for any row where P isn't one-hot.
    Fix it.
    """
    Q = np.asarray(Q, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    dO = np.asarray(dO, dtype=np.float64)
    L = np.asarray(L, dtype=np.float64)

    S = (Q @ K.T) * scale
    P = np.exp(S - L[:, None])          # recomputed attention weights

    dV = P.T @ dO
    dP = dO @ V.T

    dS = P * dP                          # BUG: missing "- D_i" correction

    dQ = (dS @ K) * scale
    dK = (dS.T @ Q) * scale
    return dQ, dK, dV
