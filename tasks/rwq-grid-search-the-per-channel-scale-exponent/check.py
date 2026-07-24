import numpy as np

RATIOS = tuple(i / 10 for i in range(11))  # 0.0, 0.1, ..., 1.0


def _quantize_symmetric_rows(Wm: np.ndarray, n_bits: int) -> np.ndarray:
    """Per-output-row symmetric round-to-nearest quantization, dequantized back."""
    qmax = 2 ** (n_bits - 1) - 1
    row_scale = np.max(np.abs(Wm), axis=1, keepdims=True) / qmax
    row_scale = np.where(row_scale == 0, 1.0, row_scale)
    q = np.round(Wm / row_scale)
    q = np.clip(q, -qmax - 1, qmax)
    return q * row_scale


def _oracle_awq_search(W: np.ndarray, X: np.ndarray, n_bits: int):
    W64 = np.asarray(W, dtype=np.float64)
    X64 = np.asarray(X, dtype=np.float64)
    Y = X64 @ W64.T  # fp baseline output on the calibration activations

    s_x = np.mean(np.abs(X64), axis=0)  # per-input-channel activation scale
    s_x = np.where(s_x == 0, 1e-12, s_x)

    mses = []
    for r in RATIOS:
        s = s_x ** r
        s = s / np.sqrt(s.max() * s.min())  # keep the scale's dynamic range balanced
        Wsc = W64 * s[None, :]
        Wq = _quantize_symmetric_rows(Wsc, n_bits)
        What = Wq / s[None, :]
        Yhat = X64 @ What.T
        mses.append(float(np.mean((Y - Yhat) ** 2)))
    mses = np.asarray(mses)
    idx = int(np.argmin(mses))
    return idx, float(mses[idx])


def grade(sol, fx) -> dict:
    """
    Builds a small weight matrix and a batch of calibration activations
    (with two salient/outlier input channels, as real calibration data has),
    and checks that the candidate's AWQ ratio-grid search finds the same
    ratio index and the same best output MSE as an oracle that enumerates
    the identical fixed ratio grid, per-channel scale, quantize, dequantize,
    and output-MSE computation.
    """
    rng = np.random.default_rng(0)
    out_f, in_f, n_samp = 12, 16, 64
    W = rng.normal(0, 1, size=(out_f, in_f)).astype(np.float32)
    X = rng.normal(0, 1, size=(n_samp, in_f)).astype(np.float32)
    X[:, 2] *= 8.0
    X[:, 7] *= 5.0
    n_bits = 4

    oracle_idx, oracle_mse = _oracle_awq_search(W, X, n_bits)

    try:
        got_idx, got_mse = sol.awq_ratio_search(W.copy(), X.copy(), n_bits)
        got_idx = int(got_idx)
        got_mse = float(got_mse)
    except Exception:
        return {"argmin_exact": 0.0, "mse_rel_err": float("inf")}

    argmin_exact = 1.0 if got_idx == oracle_idx else 0.0
    mse_rel_err = abs(got_mse - oracle_mse) / (abs(oracle_mse) + 1e-12)
    return {"argmin_exact": argmin_exact, "mse_rel_err": mse_rel_err}
