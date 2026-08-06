import math
import numpy as np

_MAG = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])


def _snap_e2m1(y):
    shape = y.shape
    flat_y = y.ravel()
    out_codes = np.empty(shape, dtype=np.float64)
    flat_out = out_codes.ravel()

    for i in range(flat_y.size.item() if hasattr(flat_y.size, 'item') else len(flat_y)):
        val = flat_y[i]
        abs_val = abs(val)
        
        best_idx = 0
        min_diff = abs(abs_val - _MAG[0])
        for j in range(1, len(_MAG)):
            diff = abs(abs_val - _MAG[j])
            if diff < min_diff:
                min_diff = diff
                best_idx = j
                
        if val < 0:
            sign_val = -1.0
        elif val > 0:
            sign_val = 1.0
        else:
            sign_val = 0.0
            
        flat_out[i] = sign_val * _MAG[best_idx]

    return out_codes


def mxfp4_quant_dequant(weights):
    x = np.asarray(weights, dtype=np.float64)
    shape = x.shape
    rows = shape[0]
    cols = shape[1]

    amax_list = []
    for i in range(rows):
        m = 0.0
        for j in range(cols):
            val = abs(x[i, j])
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

    y = np.empty(shape, dtype=np.float64)
    for i in range(rows):
        sc = scale_list[i]
        for j in range(cols):
            y[i, j] = x[i, j] / sc

    codes = _snap_e2m1(y)

    dequant = np.empty(shape, dtype=np.float64)
    for i in range(rows):
        sc = scale_list[i]
        for j in range(cols):
            dequant[i, j] = codes[i, j] * sc

    return codes, dequant
