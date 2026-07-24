import numpy as np


def _col_scale_zp(w, nbits):
    qmax = (1 << nbits) - 1
    mn = min(0.0, float(np.min(w)))
    mx = max(0.0, float(np.max(w)))
    scale = (mx - mn) / qmax if mx > mn else 1.0
    zp = int(np.clip(round(-mn / scale), 0, qmax))
    return scale, zp


def _quant_val(w, scale, zp, nbits):
    qmax = (1 << nbits) - 1
    codes = np.clip(np.round(w / scale) + zp, 0, qmax)
    return (codes - zp) * scale


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

    order = np.argsort(-np.diag(H), kind="stable")

    Hp = H[np.ix_(order, order)].copy()
    damp_val = damp * float(np.mean(np.diag(Hp)))
    Hp[np.diag_indices(d_in)] += damp_val
    Hinv = np.linalg.inv(Hp)

    scale_zp = [_col_scale_zp(W[:, c], nbits) for c in order]

    Wcur = W[:, order].copy()
    for i in range(d_in):
        w_col = Wcur[:, i]
        scale, zp = scale_zp[i]
        q_col = _quant_val(w_col, scale, zp, nbits)
        err = (w_col - q_col) / Hinv[i, i]
        Wcur[:, i] = q_col
        if i + 1 < d_in:
            Wcur[:, i + 1:] -= np.outer(err, Hinv[i, i + 1:])

    inv_order = np.argsort(order)
    Wq = Wcur[:, inv_order]
    mse = float(np.mean((Wq - W) ** 2))
    return order, mse
