import math
import numpy as np


def _prune(W, sparsity):
    Wf = W.ravel().copy()
    n_prune = int(round(sparsity * Wf.size))
    
    vals_and_indices = []
    for i in range(Wf.size):
        val = Wf[i]
        abs_val = val if val >= 0.0 else -val
        vals_and_indices.append((abs_val, i))
    
    def stable_key(item):
        return (item[0], item[1])
    
    n = len(vals_and_indices)
    for i in range(1, n):
        key = vals_and_indices[i]
        j = i - 1
        while j >= 0 and stable_key(vals_and_indices[j]) > stable_key(key):
            vals_and_indices[j + 1] = vals_and_indices[j]
            j -= 1
        vals_and_indices[j + 1] = key
        
    order = np.empty(n, dtype=np.int64)
    for i in range(n):
        order[i] = vals_and_indices[i][1]
        
    for i in range(n_prune):
        Wf[order[i]] = 0.0
        
    shape = W.shape
    res = np.empty(shape, dtype=W.dtype)
    idx = 0
    for r in range(shape[0]):
        for c in range(shape[1]):
            res[r, c] = Wf[idx]
            idx += 1
    return res


def _quantize(W, nbits):
    qmax = (1 << (nbits - 1)) - 1
    rows, cols = W.shape
    amax = np.empty(rows, dtype=W.dtype)
    for r in range(rows):
        m = 0.0
        for c in range(cols):
            val = W[r, c]
            aval = val if val >= 0.0 else -val
            if aval > m:
                m = aval
        amax[r] = m
        
    s = np.empty(rows, dtype=W.dtype)
    for r in range(rows):
        if amax[r] > 0.0:
            s[r] = amax[r] / qmax
        else:
            s[r] = 1.0
            
    codes = np.empty((rows, cols), dtype=W.dtype)
    for r in range(rows):
        sr = s[r]
        for c in range(cols):
            div = W[r, c] / sr
            if div >= 0.0:
                rounded = float(int(div + 0.5))
            else:
                rounded = float(int(div - 0.5))
            if rounded < -qmax:
                rounded = float(-qmax)
            elif rounded > qmax:
                rounded = float(qmax)
            codes[r, c] = rounded
            
    out_arr = np.empty((rows, cols), dtype=W.dtype)
    for r in range(rows):
        sr = s[r]
        for c in range(cols):
            out_arr[r, c] = codes[r, c] * sr
    return out_arr


def compound_error_bound(W: np.ndarray, X: np.ndarray, sparsity: float, nbits: int):
    """
    1. Prune W: zero the lowest-magnitude `sparsity` fraction of entries,
       globally, by stable-sorted |W| rank -> W_p.
    2. Quantize the PRUNED weights: per-row symmetric RTN at `nbits` bits
       -> W_pq.
    3. Output errors, relative to ||X W^T|| (through the linear layer):
       e_prune    = ||X W_p^T  - X W^T ||  / ||X W^T||
       e_quant    = ||X W_pq^T - X W_p^T|| / ||X W^T||   (quant error on the pruned weights)
       e_compound = ||X W_pq^T - X W^T ||  / ||X W^T||

    Returns (e_prune, e_quant, e_compound). Because
    (W_pq - W) = (W_pq - W_p) + (W_p - W) exactly, the triangle
    inequality on the Frobenius norm guarantees e_compound <= e_prune + e_quant.
    """
    W = np.asarray(W, dtype=np.float64)
    X = np.asarray(X, dtype=np.float64)

    def out(Wm):
        xr, xc = X.shape
        wr, wc = Wm.shape
        res = np.empty((xr, wr), dtype=X.dtype)
        for i in range(xr):
            for j in range(wr):
                acc = 0.0
                for k in range(xc):
                    acc += X[i, k] * Wm[j, k]
                res[i, j] = acc
        return res

    out_W = out(W)
    sum_sq_W = 0.0
    for r in range(out_W.shape[0]):
        for c in range(out_W.shape[1]):
            val = out_W[r, c]
            sum_sq_W += val * val
    denom = math.sqrt(sum_sq_W) + 1e-12

    W_p = _prune(W, sparsity)
    W_pq = _quantize(W_p, nbits)

    out_W_p = out(W_p)
    out_W_pq = out(W_pq)

    diff_prune_sum_sq = 0.0
    for r in range(out_W_p.shape[0]):
        for c in range(out_W_p.shape[1]):
            diff = out_W_p[r, c] - out_W[r, c]
            diff_prune_sum_sq += diff * diff
    e_prune = math.sqrt(diff_prune_sum_sq) / denom

    diff_quant_sum_sq = 0.0
    for r in range(out_W_pq.shape[0]):
        for c in range(out_W_pq.shape[1]):
            diff = out_W_pq[r, c] - out_W_p[r, c]
            diff_quant_sum_sq += diff * diff
    e_quant = math.sqrt(diff_quant_sum_sq) / denom

    diff_compound_sum_sq = 0.0
    for r in range(out_W_pq.shape[0]):
        for c in range(out_W_pq.shape[1]):
            diff = out_W_pq[r, c] - out_W[r, c]
            diff_compound_sum_sq += diff * diff
    e_compound = math.sqrt(diff_compound_sum_sq) / denom

    return e_prune, e_quant, e_compound
