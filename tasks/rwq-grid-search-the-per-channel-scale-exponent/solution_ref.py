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


def awq_ratio_search(W: np.ndarray, X: np.ndarray, n_bits: int = 4):
    """
    Search the fixed AWQ ratio grid RATIOS = (0.0, 0.1, ..., 1.0) for the
    ratio that minimizes the calibration-activation-weighted output MSE
    after quantizing the (scaled) weights to `n_bits`.

    For each ratio r in RATIOS:
      s_x = mean(|X|, axis=0)              # per-input-channel activation scale
      s = s_x ** r
      s = s / sqrt(s.max() * s.min())      # keep dynamic range balanced
      W_scaled = W * s[None, :]
      W_hat = dequantize(quantize_per_row(W_scaled, n_bits)) / s[None, :]
      mse = mean((X @ W.T - X @ W_hat.T) ** 2)

    Returns (best_ratio_index, best_mse): the index into RATIOS achieving
    the smallest mse, and that mse value.
    """
    W64 = np.asarray(W, dtype=np.float64)
    X64 = np.asarray(X, dtype=np.float64)
    Y = X64 @ W64.T

    s_x = np.mean(np.abs(X64), axis=0)
    s_x = np.where(s_x == 0, 1e-12, s_x)

    mses = []
    for r in RATIOS:
        s = s_x ** r
        s = s / np.sqrt(s.max() * s.min())
        Wsc = W64 * s[None, :]
        Wq = _quantize_symmetric_rows(Wsc, n_bits)
        What = Wq / s[None, :]
        Yhat = X64 @ What.T
        mses.append(float(np.mean((Y - Yhat) ** 2)))
    mses = np.asarray(mses)
    idx = int(np.argmin(mses))
    return idx, float(mses[idx])
