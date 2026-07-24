import numpy as np


def signsgd_round_step(W, scale, V, grad, lr, qmin, qmax):
    """
    Take one SignSGD step on the continuous rounding variable V:
    V_new = clip(V - lr * sign(grad), -0.5, 0.5). Then re-quantize:
    W_q = clip(round(W / scale + V_new), qmin, qmax). Return (V_new, W_q).
    """
    raise NotImplementedError('your code here')
