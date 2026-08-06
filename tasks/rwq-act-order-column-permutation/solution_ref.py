import numpy as np


def _col_scale_zp(w, nbits):
    qmax = (1 << nbits) - 1
    mn = 0.0
    mx = 0.0
    for val in w:
        fval = float(val)
        if fval < mn:
            mn = fval
        if fval > mx:
            mx = fval
    scale = (mx - mn) / qmax if mx > mn else 1.0
    val_clip = round(-mn / scale)
    if val_clip < 0:
        zp = 0
    elif val_clip > qmax:
        zp = qmax
    else:
        zp = int(val_clip)
    return scale, zp


def _quant_val(w, scale, zp, nbits):
    qmax = (1 << nbits) - 1
    res = []
    for val in w:
        fval = float(val)
        c = round(fval / scale) + zp
        if c < 0:
            c = 0
        elif c > qmax:
            c = qmax
        res.append((c - zp) * scale)
    return np.array(res, dtype=np.float64)


def _invert_matrix(A):
    n = A.shape[0]
    M = []
    for i in range(n):
        row = list(A[i]) + [1.0 if j == i else 0.0 for j in range(n)]
        M.append(row)
    
    for i in range(n):
        max_val = abs(M[i][i])
        max_row = i
        for k in range(i + 1, n):
            if abs(M[k][i]) > max_val:
                max_val = abs(M[k][i])
                max_row = k
        if max_row != i:
            M[i], M[max_row] = M[max_row], M[i]
            
        pivot = M[i][i]
        for j in range(2 * n):
            M[i][j] /= pivot
            
        for k in range(n):
            if k != i:
                factor = M[k][i]
                for j in range(2 * n):
                    M[k][j] -= factor * M[i][j]
                    
    inv_A = np.array([[M[i][n + j] for j in range(n)] for i in range(n)], dtype=np.float64)
    return inv_A


def gptq_act_order(W: np.ndarray, H: np.ndarray, nbits: int, damp: float):
    """
    Act-order = argsort(diag(H), descending). Dampen+invert the permuted
    Hessian, then run sequential GPTQ column quantization (per-column
    scale/zero-point fixed from the original W, error compensated onto the
    not-yet-quantized columns via H_inv), un-permute, and return
    (perm, mse).
    """
    W = np.asarray(W, dtype=np.float64)
    H = np.asarray(H, dtype=np.float64)
    d_out, d_in = W.shape

    order = np.array(sorted(range(d_in), key=lambda i: (-H[i, i], i)), dtype=np.int64)

    Hp = np.array([[H[order[r], order[c]] for c in range(d_in)] for r in range(d_in)], dtype=np.float64)
    
    diag_Hp_sum = 0.0
    for i in range(d_in):
        diag_Hp_sum += Hp[i, i]
    damp_val = damp * (diag_Hp_sum / d_in)
    
    for i in range(d_in):
        Hp[i, i] += damp_val
        
    Hinv = _invert_matrix(Hp)

    scale_zp = [_col_scale_zp(W[:, c], nbits) for c in order]

    Wcur = W[:, order].copy()
    for i in range(d_in):
        w_col = Wcur[:, i]
        scale, zp = scale_zp[i]
        q_col = _quant_val(w_col, scale, zp, nbits)
        err = (w_col - q_col) / Hinv[i, i]
        Wcur[:, i] = q_col
        if i + 1 < d_in:
            h_sub = Hinv[i, i + 1:]
            for r in range(d_out):
                er = err[r]
                for j_idx, c_target in enumerate(range(i + 1, d_in)):
                    Wcur[r, c_target] -= er * h_sub[j_idx]

    inv_order = np.empty(d_in, dtype=np.int64)
    for i, ord_val in enumerate(order):
        inv_order[ord_val] = i

    Wq = Wcur[:, inv_order]
    
    diff_sum = 0.0
    total_elements = d_out * d_in
    for r in range(d_out):
        for c in range(d_in):
            diff = Wq[r, c] - W[r, c]
            diff_sum += diff * diff
    mse = float(diff_sum / total_elements)
    
    return order, mse
