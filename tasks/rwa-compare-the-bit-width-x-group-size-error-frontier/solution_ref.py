def _quant_dequant_1d(x: list[float], bits: int) -> list[float]:
    n = len(x)
    qmax = (1 << bits) - 1
    if n == 0:
        return []
    xmin = float(x[0])
    xmax = float(x[0])
    for i in range(1, n):
        val = float(x[i])
        if val < xmin:
            xmin = val
        if val > xmax:
            xmax = val
    if xmax <= xmin:
        return [float(v) for v in x]
    scale = (xmax - xmin) / qmax
    zero = round(-xmin / scale)
    if zero < 0:
        zero = 0
    elif zero > qmax:
        zero = qmax
    res = [0.0] * n
    for i in range(n):
        val = float(x[i]) / scale + zero
        rounded = round(val)
        if rounded < 0:
            code = 0
        elif rounded > qmax:
            code = qmax
        else:
            code = rounded
        res[i] = (code - zero) * scale
    return res


def _grouped_dequant(W: list[list[float]], bits: int, group_size: int | None) -> list[list[float]]:
    rows = len(W)
    cols = len(W[0]) if rows > 0 else 0
    if group_size is None:
        flat = []
        for r in range(rows):
            for c in range(cols):
                flat.append(W[r][c])
        dequantized = _quant_dequant_1d(flat, bits)
        out = [[0.0] * cols for _ in range(rows)]
        idx = 0
        for r in range(rows):
            for c in range(cols):
                out[r][c] = dequantized[idx]
                idx += 1
        return out

    out = [[0.0] * cols for _ in range(rows)]
    for r in range(rows):
        for start in range(0, cols, group_size):
            end = min(start + group_size, cols)
            seg = [W[r][start + i] for i in range(end - start)]
            dequant_seg = _quant_dequant_1d(seg, bits)
            for i in range(end - start):
                out[r][start + i] = dequant_seg[i]
    return out


def bitwidth_group_mse_frontier(
    W: list[list[float]],
    bit_options: list[int],
    group_size_options: list[int | None],
) -> list[list[float]]:
    rows = len(W)
    cols = len(W[0]) if rows > 0 else 0
    num_elements = rows * cols
    mse = [[0.0] * len(group_size_options) for _ in range(len(bit_options))]
    for bi, bits in enumerate(bit_options):
        for gi, g in enumerate(group_size_options):
            W_hat = _grouped_dequant(W, bits, g)
            total_sq_err = 0.0
            for r in range(rows):
                for c in range(cols):
                    diff = W_hat[r][c] - W[r][c]
                    total_sq_err += diff * diff
            mse[bi][gi] = total_sq_err / num_elements
    return mse
