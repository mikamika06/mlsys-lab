import math


def _quantize_group_int4(W: list[list[float]], group_size: int) -> list[list[float]]:
    rows = len(W)
    cols = len(W[0]) if rows > 0 else 0
    out = [[0.0] * cols for _ in range(rows)]

    for start in range(0, cols, group_size):
        end = min(start + group_size, cols)

        max_val = 0.0
        for r in range(rows):
            for c in range(start, end):
                val = abs(W[r][c])
                if val > max_val:
                    max_val = val

        scale = max_val / 7.0
        if scale < 1e-12:
            scale = 1e-12

        for r in range(rows):
            for c in range(start, end):
                q = round(W[r][c] / scale)
                if q < -8:
                    q = -8
                elif q > 7:
                    q = 7
                out[r][c] = q * scale

    return out


def awq_vs_plain_group_int4_mse(W: list[list[float]], X: list[list[float]], group_size: int) -> tuple[float, float]:
    rows_w = len(W)
    cols_w = len(W[0]) if rows_w > 0 else 0
    rows_x = len(X)
    cols_x = len(X[0]) if rows_x > 0 else 0

    plain = _quantize_group_int4(W, group_size)

    importance = [0.0] * cols_x
    for j in range(cols_x):
        col_sum = 0.0
        for i in range(rows_x):
            col_sum += abs(X[i][j])
        importance[j] = col_sum / float(rows_x)

    imp_sum = 0.0
    for j in range(cols_x):
        imp_sum += importance[j]
    mean_imp = imp_sum / float(cols_x)

    channel_scale = [0.0] * cols_x
    denom = mean_imp + 1e-12
    for j in range(cols_x):
        channel_scale[j] = math.sqrt(importance[j] / denom)

    scaled = [[0.0] * cols_w for _ in range(rows_w)]
    for i in range(rows_w):
        for j in range(cols_w):
            scaled[i][j] = W[i][j] * channel_scale[j]

    awq_quant = _quantize_group_int4(scaled, group_size)
    awq = [[0.0] * cols_w for _ in range(rows_w)]
    for i in range(rows_w):
        for j in range(cols_w):
            awq[i][j] = awq_quant[i][j] / channel_scale[j]

    y = [[0.0] * rows_w for _ in range(rows_x)]
    for i in range(rows_x):
        for j in range(rows_w):
            acc = 0.0
            for k in range(cols_x):
                acc += X[i][k] * W[j][k]
            y[i][j] = acc

    awq_mse_sum = 0.0
    for i in range(rows_x):
        for j in range(rows_w):
            acc = 0.0
            for k in range(cols_x):
                acc += X[i][k] * awq[j][k]
            diff = y[i][j] - acc
            awq_mse_sum += diff * diff
    awq_mse = float(awq_mse_sum / float(rows_x * rows_w))

    plain_mse_sum = 0.0
    for i in range(rows_x):
        for j in range(rows_w):
            acc = 0.0
            for k in range(cols_x):
                acc += X[i][k] * plain[j][k]
            diff = y[i][j] - acc
            plain_mse_sum += diff * diff
    plain_mse = float(plain_mse_sum / float(rows_x * rows_w))

    return awq_mse, plain_mse
