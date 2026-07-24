import numpy as np


def _quant_int8_pertensor(T):
    """Per-tensor symmetric int8 quantization. Returns dequantized float."""
    scale = np.max(np.abs(T)) / 127.0 + 1e-12
    codes = np.clip(np.round(T / scale), -127, 127).astype(np.int8)
    return codes.astype(np.float32) * scale


def _ref_w8a8_errors(X, W, s):
    Y_ref = X.astype(np.float64) @ W.astype(np.float64)

    # Raw W8A8
    X_dq = _quant_int8_pertensor(X)
    W_dq = _quant_int8_pertensor(W)
    Y_raw = X_dq.astype(np.float64) @ W_dq.astype(np.float64)
    mse_raw = float(np.mean((Y_raw - Y_ref) ** 2))

    # Smoothed W8A8
    s_col = s.reshape(1, -1)
    X_hat = X / s_col
    W_hat = W * s.reshape(-1, 1)
    X_hat_dq = _quant_int8_pertensor(X_hat)
    W_hat_dq = _quant_int8_pertensor(W_hat)
    Y_smooth = X_hat_dq.astype(np.float64) @ W_hat_dq.astype(np.float64)
    mse_smooth = float(np.mean((Y_smooth - Y_ref) ** 2))

    return mse_raw, mse_smooth


def grade(sol, fx) -> dict:
    rng = np.random.default_rng(99)
    ok = True
    for _ in range(5):
        m, n, k = rng.integers(4, 16), rng.integers(4, 32), rng.integers(4, 16)
        X = rng.normal(0, 2, (m, n)).astype(np.float32)
        W = rng.normal(0, 1, (n, k)).astype(np.float32)
        # Typical SmoothQuant: s = max(|X|) along channel raised to power 0.5
        s = (np.abs(X).max(axis=0) ** 0.5 + 1e-6).astype(np.float32)

        ref_raw, ref_smooth = _ref_w8a8_errors(X, W, s)
        try:
            result = sol.w8a8_output_errors(X.copy(), W.copy(), s.copy())
            got_raw, got_smooth = float(result[0]), float(result[1])
        except Exception:
            return {"mse_check": 0.0}

        if abs(got_raw - ref_raw) > max(1e-5, 1e-4 * abs(ref_raw)):
            ok = False
            break
        if abs(got_smooth - ref_smooth) > max(1e-5, 1e-4 * abs(ref_smooth)):
            ok = False
            break

    return {"mse_check": 1.0 if ok else 0.0}
