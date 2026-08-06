import numpy as np

def sym_quant_dequant(x: np.ndarray) -> np.ndarray:
    absmax = 0.0
    for val in x.flat:
        abs_val = val if val >= 0 else -val
        if abs_val > absmax:
            absmax = abs_val

    scale = absmax / 127 if absmax != 0 else 1.0

    out = np.empty(x.shape, dtype=np.float64)
    it = np.nditer(x, flags=['multi_index'])
    while not it.finished:
        val = it[0]
        q = round(val / scale)
        if q < -128:
            q = -128
        elif q > 127:
            q = 127
        q_int8 = int(q)
        out[it.multi_index] = float(q_int8) * scale
        it.iternext()

    return out

def asym_quant_dequant(x: np.ndarray) -> np.ndarray:
    mn = float('inf')
    mx = float('-inf')
    for val in x.flat:
        if val < mn:
            mn = val
        if val > mx:
            mx = val

    rng = mx - mn
    if rng == 0:
        scale = 1.0
        zp = 128
    else:
        scale = rng / 255
        zp = int(round(-mn / scale))

    out = np.empty(x.shape, dtype=np.float64)
    it = np.nditer(x, flags=['multi_index'])
    while not it.finished:
        val = it[0]
        q = round(val / scale + zp)
        if q < 0:
            q = 0
        elif q > 255:
            q = 255
        q_uint8 = int(q)
        out[it.multi_index] = (float(q_uint8) - zp) * scale
        it.iternext()

    return out
