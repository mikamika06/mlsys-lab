import numpy as np


def make_qx_quants(x, w, nmax):
    x = np.asarray(x, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    
    shape = x.shape
    x_flat = x.ravel()
    w_flat = w.ravel()
    n = x_flat.shape[0]
    
    amax = 0.0
    for i in range(n):
        val = x_flat[i]
        abs_val = val if val >= 0.0 else -val
        if abs_val > amax:
            amax = abs_val
            
    if amax == 0:
        return -1, np.zeros(shape, dtype=np.int64)

    base_scale = amax / nmax
    best_idx = -1
    best_err = None
    best_codes = None
    
    for k in range(-15, 16):
        idx = k + 15
        scale = base_scale * (1.0 + k / 32.0)
        if scale == 0:
            continue
            
        codes_list = []
        err = 0.0
        for i in range(n):
            xi = x_flat[i]
            wi = w_flat[i]
            r = round(xi / scale)
            if r < -nmax:
                c = -nmax
            elif r > nmax:
                c = nmax
            else:
                c = r
            c_float = float(c)
            codes_list.append(c_float)
            diff = xi - scale * c_float
            err += wi * (diff * diff)
            
        if best_err is None or err < best_err:
            best_err = err
            best_idx = idx
            best_codes = np.array(codes_list, dtype=np.int64).reshape(shape)
            
    return best_idx, best_codes
