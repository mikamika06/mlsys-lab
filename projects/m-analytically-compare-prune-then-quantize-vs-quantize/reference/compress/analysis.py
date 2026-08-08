import numpy as np


def analytical_error(W, mask, scale, zero_point):
    w_ptq = np.where(mask == 0, 0.0, W)
    q_ptq = np.clip(np.round(w_ptq / scale) + zero_point, -8, 7)
    deq_ptq = (q_ptq - zero_point) * scale
    err_ptq = np.mean((W - deq_ptq) ** 2)

    q_qp = np.clip(np.round(W / scale) + zero_point, -8, 7)
    deq_qp = (q_qp - zero_point) * scale
    w_qp = np.where(mask == 0, 0.0, deq_qp)
    err_qp = np.mean((W - w_qp) ** 2)

    return float(err_ptq), float(err_qp)
