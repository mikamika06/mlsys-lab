import numpy as np

from mlsys import scorers


def _int8_roundtrip(x):
    amax = float(np.max(np.abs(x)))
    scale = max(amax / 127.0, 1e-12)
    q = np.clip(np.round(x / scale), -127, 127)
    return q * scale


def _oracle(X, W, alpha):
    Y_true = X @ W.T

    X_hat_raw = _int8_roundtrip(X)
    W_hat_raw = _int8_roundtrip(W)
    Y_raw = X_hat_raw @ W_hat_raw.T
    error_raw = float(np.linalg.norm(Y_raw - Y_true) / np.linalg.norm(Y_true))

    x_amax = np.max(np.abs(X), axis=0)
    w_amax = np.max(np.abs(W), axis=0)
    s = (x_amax ** alpha) / np.maximum(w_amax ** (1.0 - alpha), 1e-12)
    s = np.maximum(s, 1e-12)

    X_smooth = X / s[None, :]
    W_smooth = W * s[None, :]
    X_hat_sm = _int8_roundtrip(X_smooth)
    W_hat_sm = _int8_roundtrip(W_smooth)
    Y_smooth = X_hat_sm @ W_hat_sm.T
    error_smoothed = float(np.linalg.norm(Y_smooth - Y_true) / np.linalg.norm(Y_true))

    ratio = error_smoothed / error_raw if error_raw > 0 else float("inf")
    return error_raw, error_smoothed, ratio


def _synthetic_cases():
    rng = np.random.default_rng(113)
    cases = []
    for _ in range(4):
        n = int(rng.integers(20, 60))
        d_in = int(rng.integers(6, 20))
        d_out = int(rng.integers(4, 16))
        X = rng.standard_normal((n, d_in)) * 0.5
        n_outliers = int(rng.integers(1, min(4, d_in) + 1))
        outlier_channels = rng.choice(d_in, size=n_outliers, replace=False)
        X[:, outlier_channels] *= rng.uniform(8.0, 25.0, size=n_outliers)
        W = rng.standard_normal((d_out, d_in)) * 0.3
        alpha = float(rng.uniform(0.3, 0.7))
        cases.append((X, W, alpha))
    return cases


def grade(sol, fx) -> dict:
    cases = [(fx["X"], fx["W"], 0.5)] + _synthetic_cases()

    worst = 0.0
    for X, W, alpha in cases:
        ref_raw, ref_sm, ref_ratio = _oracle(X, W, alpha)
        try:
            got = sol.smoothquant_w8a8_comparison(X.copy(), W.copy(), alpha)
            got_raw = float(got["error_raw"])
            got_sm = float(got["error_smoothed"])
            got_ratio = float(got["improvement_ratio"])
        except Exception:
            return {"rel_err": float("inf")}

        ref_vec = np.array([ref_raw, ref_sm, ref_ratio])
        got_vec = np.array([got_raw, got_sm, got_ratio])
        err = scorers.rel_err(ref_vec, got_vec)
        worst = max(worst, err)

    return {"rel_err": worst}
