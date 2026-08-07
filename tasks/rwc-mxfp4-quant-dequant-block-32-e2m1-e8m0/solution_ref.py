import math

_MAG = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]


def _snap_e2m1(y: list[list[float]]) -> list[list[float]]:
    rows = len(y)
    cols = len(y[0]) if rows > 0 else 0
    out_codes = []

    for i in range(rows):
        row_codes = []
        for j in range(cols):
            val = y[i][j]
            abs_val = abs(val)

            best_idx = 0
            min_diff = abs(abs_val - _MAG[0])
            for k in range(1, len(_MAG)):
                diff = abs(abs_val - _MAG[k])
                if diff < min_diff:
                    min_diff = diff
                    best_idx = k

            if val < 0:
                sign_val = -1.0
            elif val > 0:
                sign_val = 1.0
            else:
                sign_val = 0.0

            row_codes.append(sign_val * _MAG[best_idx])
        out_codes.append(row_codes)

    return out_codes


def mxfp4_quant_dequant(weights: list[list[float]]) -> tuple[list[list[float]], list[list[float]]]:
    x = weights
    rows = len(x)
    cols = len(x[0]) if rows > 0 else 0

    amax_list = []
    for i in range(rows):
        m = 0.0
        for j in range(cols):
            val = abs(x[i][j])
            if val > m:
                m = val
        amax_list.append(m)

    ratio_list = []
    for m in amax_list:
        if m > 0.0:
            ratio_list.append(m / 6.0)
        else:
            ratio_list.append(6.0 / 6.0)

    e_list = []
    for r in ratio_list:
        if r > 0.0:
            val_log = math.log2(r)
            val_ceil = math.ceil(val_log)
            if val_ceil > 0:
                e_list.append(int(val_ceil))
            else:
                e_list.append(0)
        else:
            e_list.append(0)

    scale_list = []
    for e in e_list:
        scale_list.append(2.0 ** float(e))

    y = []
    for i in range(rows):
        sc = scale_list[i]
        row_y = []
        for j in range(cols):
            row_y.append(x[i][j] / sc)
        y.append(row_y)

    codes = _snap_e2m1(y)

    dequant = []
    for i in range(rows):
        sc = scale_list[i]
        row_dequant = []
        for j in range(cols):
            row_dequant.append(codes[i][j] * sc)
        dequant.append(row_dequant)

    return codes, dequant
