def stable_log1p(x):
    import numpy as np
    import math
    x_arr = np.asarray(x, dtype=np.float64)
    shape = x_arr.shape
    x_flat = x_arr.ravel()
    out = np.empty(x_flat.size, dtype=np.float64)
    for i in range(x_flat.size):
        v = float(x_flat[i])
        u = 1.0 + v
        if u == 1.0:
            out[i] = v
        elif u == 0.0:
            out[i] = float("-inf")
        elif u < 0.0:
            out[i] = float("nan")
        else:
            out[i] = math.log(u) * (v / (u - 1.0))
    return out.reshape(shape)

def stable_expm1(x):
    import numpy as np
    import math
    x_arr = np.asarray(x, dtype=np.float64)
    shape = x_arr.shape
    x_flat = x_arr.ravel()
    out = np.empty(x_flat.size, dtype=np.float64)
    for i in range(x_flat.size):
        v = float(x_flat[i])
        if math.isnan(v):
            out[i] = float("nan")
            continue
        try:
            u = math.exp(v)
        except OverflowError:
            out[i] = float("inf")
            continue
        if u == 1.0:
            out[i] = v
        elif u == 0.0:
            out[i] = -1.0
        else:
            out[i] = (u - 1.0) * (v / math.log(u))
    return out.reshape(shape)
