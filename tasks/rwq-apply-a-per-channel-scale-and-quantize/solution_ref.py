import math


def _qd_1d(x: list[float], bits: int) -> list[float]:
    qmax = (1 << bits) - 1

    xmin = x[0]
    xmax = x[0]
    for val in x:
        if val < xmin:
            xmin = val
        if val > xmax:
            xmax = val

    if xmax <= xmin:
        return list(x)

    scale = (xmax - xmin) / qmax
    zero = round(-xmin / scale)
    if zero < 0:
        zero = 0
    if zero > qmax:
        zero = qmax

    n = len(x)
    codes = [0.0] * n
    for i in range(n):
        val = round(x[i] / scale) + zero
        if val < 0:
            val = 0
        if val > qmax:
            val = qmax
        codes[i] = val

    out = [0.0] * n
    for i in range(n):
        out[i] = (codes[i] - zero) * scale
    return out


def _group_quant_rows(W: list[list[float]], group_size: int, bits: int) -> list[list[float]]:
    rows = len(W)
    cols = len(W[0])
    out = [[0.0] * cols for _ in range(rows)]
    for r in range(rows):
        row = W[r]
        for c0 in range(0, cols, group_size):
            c_end = c0 + group_size
            if c_end > cols:
                c_end = cols
            seg = row[c0:c_end]
            res = _qd_1d(seg, bits)
            for i, val in enumerate(res):
                out[r][c0 + i] = val
    return out


def awq_scale_and_quantize(W: list[list[float]], X: list[list[float]], s: list[float], group_size: int, bits: int = 4) -> tuple[list[list[float]], list[list[float]]]:
    rows_w = len(W)
    cols_w = len(W[0])
    Wp = [[0.0] * cols_w for _ in range(rows_w)]
    for r in range(rows_w):
        for c in range(cols_w):
            Wp[r][c] = W[r][c] * s[c]

    rows_x = len(X)
    cols_x = len(X[0])
    Xp = [[0.0] * cols_x for _ in range(rows_x)]
    for r in range(rows_x):
        for c in range(cols_x):
            Xp[r][c] = X[r][c] / s[c]

    Y_identity = [[0.0] * rows_w for _ in range(rows_x)]
    for i in range(rows_x):
        for j in range(rows_w):
            acc = 0.0
            for k in range(cols_x):
                acc += Xp[i][k] * Wp[j][k]
            Y_identity[i][j] = acc

    W_hat = _group_quant_rows(Wp, group_size, bits)
    Y_quant = [[0.0] * rows_w for _ in range(rows_x)]
    for i in range(rows_x):
        for j in range(rows_w):
            acc = 0.0
            for k in range(cols_x):
                acc += Xp[i][k] * W_hat[j][k]
            Y_quant[i][j] = acc

    return Y_identity, Y_quant
