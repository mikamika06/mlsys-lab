import math
import numpy as np


def _qd_1d(x, bits):
    x = np.asarray(x, dtype=np.float64)
    qmax = (1 << bits) - 1
    
    xmin = x[0]
    xmax = x[0]
    for val in x:
        if val < xmin:
            xmin = val
        if val > xmax:
            xmax = val
            
    if xmax <= xmin:
        return x.copy()
        
    scale = (xmax - xmin) / qmax
    zero = round(-xmin / scale)
    if zero < 0:
        zero = 0
    if zero > qmax:
        zero = qmax
        
    n = len(x)
    codes = np.empty(n, dtype=np.float64)
    for i in range(n):
        val = round(x[i] / scale) + zero
        if val < 0:
            val = 0
        if val > qmax:
            val = qmax
        codes[i] = val
        
    out = np.empty(n, dtype=np.float64)
    for i in range(n):
        out[i] = (codes[i] - zero) * scale
    return out


def _group_quant_rows(W, group_size, bits):
    rows, cols = W.shape
    out = np.empty((rows, cols), dtype=np.float64)
    for r in range(rows):
        row = W[r]
        for c0 in range(0, cols, group_size):
            c_end = c0 + group_size
            if c_end > cols:
                c_end = cols
            seg = row[c0:c_end]
            res = _qd_1d(seg, bits)
            for i, val in enumerate(res):
                out[r, c0 + i] = val
    return out


def awq_scale_and_quantize(W, X, s, group_size, bits=4):
    W = np.asarray(W, dtype=np.float64)
    X = np.asarray(X, dtype=np.float64)
    s = np.asarray(s, dtype=np.float64)
    
    rows_w, cols_w = W.shape
    Wp = np.empty((rows_w, cols_w), dtype=np.float64)
    for r in range(rows_w):
        for c in range(cols_w):
            Wp[r, c] = W[r, c] * s[c]
            
    rows_x, cols_x = X.shape
    Xp = np.empty((rows_x, cols_x), dtype=np.float64)
    for r in range(rows_x):
        for c in range(cols_x):
            Xp[r, c] = X[r, c] / s[c]
            
    Wp_t = Wp.T
    cols_wp_t = Wp_t.shape[1]
    Y_identity = np.empty((rows_x, cols_wp_t), dtype=np.float64)
    for i in range(rows_x):
        for j in range(cols_wp_t):
            acc = 0.0
            for k in range(cols_x):
                acc += Xp[i, k] * Wp_t[k, j]
            Y_identity[i, j] = acc
            
    W_hat = _group_quant_rows(Wp, group_size, bits)
    W_hat_t = W_hat.T
    cols_what_t = W_hat_t.shape[1]
    Y_quant = np.empty((rows_x, cols_what_t), dtype=np.float64)
    for i in range(rows_x):
        for j in range(cols_what_t):
            acc = 0.0
            for k in range(cols_x):
                acc += Xp[i, k] * W_hat_t[k, j]
            Y_quant[i, j] = acc

    return Y_identity, Y_quant
