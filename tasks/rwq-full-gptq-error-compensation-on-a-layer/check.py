import numpy as np

from mlsys import scorers

NBITS = 3
DAMP = 0.01


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


def _gptq(W, H, nbits, damp):
    """Natural left-to-right column order, H^-1 error compensation."""
    W = np.asarray(W, dtype=np.float64).copy()
    d_out, d_in = W.shape
    Hp = H.astype(np.float64).copy()
    damp_val = damp * float(np.mean(np.diag(Hp)))
    Hp[np.diag_indices(d_in)] += damp_val
    Hinv = np.linalg.inv(Hp)

    scale_zp = [_col_scale_zp(W[:, c], nbits) for c in range(d_in)]

    for i in range(d_in):
        w_col = W[:, i]
        scale, zp = scale_zp[i]
        q_col = _quant_val(w_col, scale, zp, nbits)
        err = (w_col - q_col) / Hinv[i, i]
        W[:, i] = q_col
        if i + 1 < d_in:
            W[:, i + 1:] -= np.outer(err, Hinv[i, i + 1:])
    return W


def _rtn(W, nbits):
    """Plain round-to-nearest, no error compensation (the baseline)."""
    W = np.asarray(W, dtype=np.float64).copy()
    d_out, d_in = W.shape
    for c in range(d_in):
        scale, zp = _col_scale_zp(W[:, c], nbits)
        W[:, c] = _quant_val(W[:, c], scale, zp, nbits)
    return W


def _out_mse(Wq, X, W):
    Yh = X @ Wq.T
    Y = X @ W.T
    return float(np.mean((Yh - Y) ** 2))


def grade(sol, fx) -> dict:
    """
    Loads the fixed calibration weight/activation fixtures, runs the
    reference full-GPTQ (natural column order, H^-1 error compensation)
    and the RTN baseline with a NumPy oracle, and compares:
    - the submission's quantized weight matrix (max abs error) to the oracle,
    - the submission's reported layer-output MSE (relative error) to the
      oracle's, and
    - whether the submission's MSE actually beats the RTN baseline's MSE.
    """
    W = np.asarray(fx["gptq_w"], dtype=np.float64)
    X = np.asarray(fx["gptq_x"], dtype=np.float64)
    H = X.T @ X

    Wq_exp = _gptq(W, H, NBITS, DAMP)
    mse_exp = _out_mse(Wq_exp, X, W)
    mse_rtn = _out_mse(_rtn(W, NBITS), X, W)

    try:
        Wq_got, mse_got = sol.gptq_quantize_layer(W.copy(), X.copy(), NBITS, DAMP)
        Wq_got = np.asarray(Wq_got, dtype=np.float64)
        mse_got = float(mse_got)
    except Exception:
        return {
            "wq_max_abs_err": float("inf"),
            "mse_rel_err": float("inf"),
            "beats_rtn": 0.0,
        }

    if Wq_got.shape != Wq_exp.shape:
        wq_max_abs_err = float("inf")
    else:
        wq_max_abs_err = scorers.max_abs_err(Wq_exp, Wq_got)

    mse_rel_err = abs(mse_got - mse_exp) / (abs(mse_exp) + 1e-12)
    beats_rtn = 1.0 if mse_got < mse_rtn else 0.0

    return {
        "wq_max_abs_err": wq_max_abs_err,
        "mse_rel_err": mse_rel_err,
        "beats_rtn": beats_rtn,
    }
