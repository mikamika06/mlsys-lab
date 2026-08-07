def _qd_1d(x, bits=4):
    qmax = (1 << bits) - 1

    n = len(x)
    if n == 0:
        return list(x)

    xmin = float(x[0])
    xmax = float(x[0])
    for i in range(1, n):
        val = float(x[i])
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
    elif zero > qmax:
        zero = qmax

    out = [0.0] * n
    for i in range(n):
        val = float(x[i])
        val_code = round(val / scale + zero)
        if val_code < 0:
            c = 0
        elif val_code > qmax:
            c = qmax
        else:
            c = val_code
        out[i] = (c - zero) * scale

    return out


def quantize_dequantize_int4_grouped(x: list[list[float]], group_size: int) -> list[list[float]]:
    rows = len(x)
    if rows == 0:
        return []
    cols = len(x[0])
    out = [[0.0] * cols for _ in range(rows)]
    for r in range(rows):
        row = x[r]
        for s in range(0, cols, group_size):
            seg = row[s:s + group_size]
            seg_q = _qd_1d(seg, bits=4)
            for i, val in enumerate(seg_q):
                out[r][s + i] = val
    return out
