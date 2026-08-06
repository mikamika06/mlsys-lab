import numpy as np


def roofline_attainable(flops, bytes_moved, peak_flops, bandwidth):
    """Roofline: arithmetic intensity, attainable FLOP/s, ridge point."""
    W = np.asarray(flops, dtype=np.float64)
    Q = np.asarray(bytes_moved, dtype=np.float64)
    P = float(peak_flops)
    B = float(bandwidth)

    n = W.shape[0]
    ai_list = []
    attainable_list = []

    for i in range(n):
        w_val = W[i]
        q_val = Q[i]
        ai_val = w_val / q_val
        ai_list.append(ai_val)

        prod = ai_val * B
        if P < prod:
            att_val = P
        else:
            att_val = prod
        attainable_list.append(att_val)

    ai = np.array(ai_list, dtype=np.float64)
    attainable = np.array(attainable_list, dtype=np.float64)
    ridge = P / B
    return ai, attainable, ridge
