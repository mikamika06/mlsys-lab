import numpy as np

def asymmetric_quant_round_trip(x):
    x = np.asarray(x, dtype=np.float64)
    min_val = x.min()
    max_val = x.max()
    if max_val == min_val:
        s = 1.0
    else:
        s = (max_val - min_val) / 255.0
    z_float = -min_val / s
    zp = int(round(z_float))
    zp = np.clip(zp, -128, 127)
    q = np.round(x / s + zp).astype(np.int32)
    q = np.clip(q, -128, 127)
    deq = (q.astype(np.float64) - zp) * s
    return deq, int(zp)
