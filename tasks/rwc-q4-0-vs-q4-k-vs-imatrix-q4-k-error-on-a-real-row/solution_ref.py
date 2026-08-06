import math
import numpy as np


def _wmse(x, x_hat, w):
    n = x.shape[0]
    num = 0.0
    den = 0.0
    for i in range(n):
        diff = x[i] - x_hat[i]
        num += w[i] * (diff * diff)
        den += w[i]
    return float(num / den)


def _q4_0(x):
    n = x.shape[0]
    amax = 0.0
    for i in range(n):
        val = x[i]
        if val < 0.0:
            val = -val
        if val > amax:
            amax = val
    
    d = amax / 8.0 if amax != 0.0 else 1e-12
    
    codes = np.empty(n, dtype=np.float64)
    recon = np.empty(n, dtype=np.float64)
    for i in range(n):
        c = round(x[i] / d)
        if c < -8.0:
            c = -8.0
        elif c > 7.0:
            c = 7.0
        codes[i] = c
        recon[i] = d * c
    return recon


def _search_scale(x, weight):
    n = x.shape[0]
    amax = 0.0
    for i in range(n):
        val = x[i]
        if val < 0.0:
            val = -val
        if val > amax:
            amax = val
            
    d0 = amax / 8.0 if amax != 0.0 else 1e-12
    best_err = None
    best_recon = None
    
    for k in range(-15, 16):
        d = d0 * (1.0 + k / 32.0)
        if d == 0.0:
            continue
            
        recon = np.empty(n, dtype=np.float64)
        err = 0.0
        for i in range(n):
            c = round(x[i] / d)
            if c < -8.0:
                c = -8.0
            elif c > 7.0:
                c = 7.0
            r = d * c
            recon[i] = r
            diff = x[i] - r
            err += weight[i] * (diff * diff)
            
        if best_err is None or err < best_err:
            best_err = err
            best_recon = recon
            
    return best_recon


def compare_q4_variants(x: np.ndarray, w: np.ndarray) -> tuple:
    x = np.asarray(x, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)

    recon_q4_0 = _q4_0(x)
    
    n = w.shape[0]
    ones_w = np.empty(n, dtype=np.float64)
    for i in range(n):
        ones_w[i] = 1.0
        
    recon_q4_k = _search_scale(x, ones_w)
    recon_imatrix = _search_scale(x, w)

    err0 = _wmse(x, recon_q4_0, w)
    err1 = _wmse(x, recon_q4_k, w)
    err2 = _wmse(x, recon_imatrix, w)

    errors = np.array([err0, err1, err2], dtype=np.float64)
    
    best_idx = 0
    min_val = errors[0]
    if errors[1] < min_val:
        min_val = errors[1]
        best_idx = 1
    if errors[2] < min_val:
        min_val = errors[2]
        best_idx = 2
        
    return errors, int(best_idx)
