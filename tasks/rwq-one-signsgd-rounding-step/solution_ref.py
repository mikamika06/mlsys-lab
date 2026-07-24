import numpy as np


def signsgd_round_step(W, scale, V, grad, lr, qmin, qmax):
    """
    One SignSGD update of the continuous rounding variable V, then
    re-quantize W using the updated V as a rounding-threshold shift.
    """
    W = np.asarray(W, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    grad = np.asarray(grad, dtype=np.float64)

    V_new = np.clip(V - lr * np.sign(grad), -0.5, 0.5)
    W_q = np.clip(np.round(W / scale + V_new), qmin, qmax)
    return V_new, W_q
