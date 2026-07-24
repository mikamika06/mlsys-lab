import numpy as np

from mlsys import scorers

FAIL = {
    "rtn_rel_err": float("inf"),
    "gptq_rel_err": float("inf"),
    "gptq_over_rtn_output_error": float("inf"),
    "rtn_output_rel_err": float("inf"),
    "gptq_output_rel_err": float("inf"),
}


# ---- oracle (recomputed here with numpy, never hardcoded) -------------------
def _row_scales(W, bits):
    qmax = 2 ** (bits - 1) - 1
    scale = np.max(np.abs(W), axis=1) / qmax
    scale = np.where(scale == 0.0, 1.0, scale)
    return scale, qmax


def _ref_rtn(W, bits):
    W = np.asarray(W, dtype=np.float64)
    scale, qmax = _row_scales(W, bits)
    return np.clip(np.rint(W / scale[:, None]), -qmax, qmax) * scale[:, None]


def _ref_gptq(W, X, bits, damp=0.01):
    W = np.asarray(W, dtype=np.float64)
    X = np.asarray(X, dtype=np.float64)
    d_in = W.shape[1]
    H = X.T @ X
    H = H + damp * float(np.mean(np.diag(H))) * np.eye(d_in)
    scale, qmax = _row_scales(W, bits)
    U = np.linalg.cholesky(np.linalg.inv(H)).T
    Wc = W.copy()
    Q = np.zeros_like(W)
    for j in range(d_in):
        w = Wc[:, j]
        q = np.clip(np.rint(w / scale), -qmax, qmax) * scale
        Q[:, j] = q
        err = (w - q) / U[j, j]
        if j + 1 < d_in:
            Wc[:, j + 1:] -= np.outer(err, U[j, j + 1:])
    return Q


def _cases(fx):
    out = []
    W = np.asarray(fx["W"], dtype=np.float64)
    X = np.asarray(fx["X"], dtype=np.float64)
    out.append((W, X, 3))
    out.append((W, X, 4))

    rng = np.random.default_rng(0)
    d_in, d_out, n_cal = 24, 16, 128
    mix = rng.normal(size=(d_in, d_in)) / np.sqrt(d_in)
    X2 = rng.normal(size=(n_cal, d_in)) @ mix
    W2 = rng.normal(size=(d_out, d_in))
    out.append((W2, X2, 3))
    return out


def grade(sol, fx) -> dict:
    worst_rtn = 0.0
    worst_gptq = 0.0
    ratios = []
    out_rtn = []
    out_gptq = []

    for W, X, bits in _cases(fx):
        ref_rtn = _ref_rtn(W, bits)
        ref_gptq = _ref_gptq(W, X, bits)

        try:
            got_rtn = np.asarray(sol.quantize_rtn(W.copy(), bits), dtype=np.float64)
        except Exception:
            return dict(FAIL)
        try:
            got_gptq = np.asarray(sol.quantize_gptq(W.copy(), X.copy(), bits), dtype=np.float64)
        except Exception:
            return dict(FAIL)

        if got_rtn.shape != W.shape or got_gptq.shape != W.shape:
            return dict(FAIL)
        if not np.all(np.isfinite(got_rtn)) or not np.all(np.isfinite(got_gptq)):
            return dict(FAIL)

        worst_rtn = max(worst_rtn, scorers.rel_err(ref_rtn, got_rtn))
        worst_gptq = max(worst_gptq, scorers.rel_err(ref_gptq, got_gptq))

        Y = X @ W.T
        e_rtn = scorers.rel_err(Y, X @ got_rtn.T)
        e_gptq = scorers.rel_err(Y, X @ got_gptq.T)
        out_rtn.append(e_rtn)
        out_gptq.append(e_gptq)
        ratios.append(e_gptq / e_rtn if e_rtn > 0 else float("inf"))

    return {
        "rtn_rel_err": float(worst_rtn),
        "gptq_rel_err": float(worst_gptq),
        "gptq_over_rtn_output_error": float(max(ratios)),
        "rtn_output_rel_err": float(np.mean(out_rtn)),
        "gptq_output_rel_err": float(np.mean(out_gptq)),
    }
