import math
import numpy as np


def swiglu_ffn(x, gate_w, up_w, down_w):
    x_arr = np.asarray(x, dtype=np.float64)
    gw_arr = np.asarray(gate_w, dtype=np.float64)
    uw_arr = np.asarray(up_w, dtype=np.float64)
    dw_arr = np.asarray(down_w, dtype=np.float64)

    m = x_arr.shape[0]
    k_gate = x_arr.shape[1]
    n_gate = gw_arr.shape[1]

    gate = [[0.0] * n_gate for _ in range(m)]
    for i in range(m):
        for j in range(n_gate):
            s = 0.0
            for k in range(k_gate):
                s += x_arr[i, k] * gw_arr[k, j]
            gate[i][j] = s

    n_up = uw_arr.shape[1]
    up = [[0.0] * n_up for _ in range(m)]
    for i in range(m):
        for j in range(n_up):
            s = 0.0
            for k in range(k_gate):
                s += x_arr[i, k] * uw_arr[k, j]
            up[i][j] = s

    hidden = [[0.0] * n_gate for _ in range(m)]
    for i in range(m):
        for j in range(n_gate):
            g = gate[i][j]
            silu = g / (1.0 + math.exp(-g))
            hidden[i][j] = silu * up[i][j]

    n_down = dw_arr.shape[1]
    k_down = dw_arr.shape[0]
    result = [[0.0] * n_down for _ in range(m)]
    for i in range(m):
        for j in range(n_down):
            s = 0.0
            for p in range(k_down):
                s += hidden[i][p] * dw_arr[p, j]
            result[i][j] = s

    return np.asarray(result, dtype=np.float64)
