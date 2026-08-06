import numpy as np


def obq_column_step(W, H_inv, col, scale, nmax):
    W = np.asarray(W, dtype=np.float64).copy()
    H_inv = np.asarray(H_inv, dtype=np.float64)
    scale = np.asarray(scale, dtype=np.float64)

    rows = W.shape[0]
    n = W.shape[1]

    q_col_list = []
    err_list = []
    h_col_col = H_inv[col, col]

    for i in range(rows):
        w_val = W[i, col]
        s_val = scale[i]
        div = w_val / s_val
        rounded = round(div)
        if rounded < -nmax:
            code = -nmax
        elif rounded > nmax:
            code = nmax
        else:
            code = rounded
        q_val = code * s_val
        q_col_list.append(q_val)
        e_val = (w_val - q_val) / h_col_col
        err_list.append(e_val)
        W[i, col] = q_val

    if col + 1 < n:
        for i in range(rows):
            e = err_list[i]
            for j in range(col + 1, n):
                W[i, j] -= e * H_inv[col, j]

    q_col = np.asarray(q_col_list, dtype=np.float64)
    return q_col, W
