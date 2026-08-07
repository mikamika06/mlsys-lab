def _qd_1d(x: list[float], bits: int) -> list[float]:
    qmax = (1 << bits) - 1
    xmin = float("inf")
    xmax = float("-inf")
    for val in x:
        if val < xmin:
            xmin = val
        if val > xmax:
            xmax = val
    if xmax <= xmin:
        return list(x)
    scale = (xmax - xmin) / qmax
    zero = round(-xmin / scale)
    zero = min(max(zero, 0), qmax)
    out_list = []
    for val in x:
        code = round(val / scale) + zero
        if code < 0:
            code = 0
        elif code > qmax:
            code = qmax
        out_list.append((code - zero) * scale)
    return out_list


def _group_quant(x: list[list[float]], axis: int, group_size: int, bits: int) -> list[list[float]]:
    rows = len(x)
    cols = len(x[0]) if rows > 0 else 0
    out = [[0.0] * cols for _ in range(rows)]
    if axis == 0:
        for c in range(cols):
            col = [x[r][c] for r in range(rows)]
            for s in range(0, rows, group_size):
                seg = col[s:s + group_size]
                res = _qd_1d(seg, bits)
                for i, val in enumerate(res):
                    out[s + i][c] = val
    else:
        for r in range(rows):
            row = x[r]
            for s in range(0, cols, group_size):
                seg = row[s:s + group_size]
                res = _qd_1d(seg, bits)
                for i, val in enumerate(res):
                    out[r][s + i] = val
    return out


def quantize_dequantize_kv(K: list[list[float]], V: list[list[float]], group_size: int, bits: int = 4) -> tuple[list[list[float]], list[list[float]]]:
    K_hat = _group_quant(K, axis=0, group_size=group_size, bits=bits)
    V_hat = _group_quant(V, axis=1, group_size=group_size, bits=bits)
    return K_hat, V_hat
