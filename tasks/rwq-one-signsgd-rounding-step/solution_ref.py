import numpy as np


def signsgd_round_step(W, scale, V, grad, lr, qmin, qmax):
    """
    One SignSGD update of the continuous rounding variable V, then
    re-quantize W using the updated V as a rounding-threshold shift.
    """
    W = np.asarray(W, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    grad = np.asarray(grad, dtype=np.float64)

    n = W.shape[0]
    V_new_list = []
    W_q_list = []

    for i in range(n):
        g = grad[i]
        if g > 0.0:
            s = 1.0
        elif g < 0.0:
            s = -1.0
        else:
            s = 0.0

        v_val = V[i] - lr * s
        if v_val < -0.5:
            v_new = -0.5
        elif v_val > 0.5:
            v_new = 0.5
        else:
            v_new = v_val
        V_new_list.append(v_new)

        w_val = W[i] / scale + v_new
        r_val = float(round(w_val))
        if r_val < qmin:
            wq = float(qmin)
        elif r_val > qmax:
            wq = float(qmax)
        else:
            wq = r_val
        W_q_list.append(wq)

    V_new = np.asarray(V_new_list, dtype=np.float64)
    W_q = np.asarray(W_q_list, dtype=np.float64)
    return V_new, W_q
