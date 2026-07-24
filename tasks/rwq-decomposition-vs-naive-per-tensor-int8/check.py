import numpy as np

def _oracle(X, threshold=6.0):
    """Reference oracle: compute MSE for both quantization strategies."""
    X = np.asarray(X, dtype=np.float64)

    # --- Naive per-tensor int8 ---
    abs_max = np.max(np.abs(X))
    scale_n = abs_max / 127.0 if abs_max > 0 else 1.0
    Xq_n = np.clip(np.round(X / scale_n), -128, 127).astype(np.int8)
    Xr_n = Xq_n.astype(np.float64) * scale_n
    mse_naive = float(np.mean((X - Xr_n) ** 2))

    # --- LLM.int8() decomposition ---
    col_max = np.max(np.abs(X), axis=0)
    outlier = col_max > threshold
    Xr = np.empty_like(X)

    # Non-outlier columns -> int8
    if np.any(~outlier):
        Xsub = X[:, ~outlier]
        s = np.max(np.abs(Xsub))
        s = s / 127.0 if s > 0 else 1.0
        Xq = np.clip(np.round(Xsub / s), -128, 127).astype(np.int8)
        Xr[:, ~outlier] = Xq.astype(np.float64) * s

    # Outlier columns -> fp16
    if np.any(outlier):
        Xr[:, outlier] = X[:, outlier].astype(np.float16).astype(np.float64)

    mse_decomp = float(np.mean((X - Xr) ** 2))
    return mse_decomp, mse_naive

def _make_fixture():
    """Deterministic outlier-heavy matrix."""
    rng = np.random.RandomState(42)
    X = rng.randn(64, 32) * 0.5
    X[:, 3] = rng.randn(64) * 100
    X[:, 17] = rng.randn(64) * 50
    X[:, 28] = rng.randn(64) * 80
    return X

def grade(sol, fx) -> dict:
    X = _make_fixture()

    ref_d, ref_n = _oracle(X)

    try:
        got_d, got_n = sol.compare_quantization(X.copy())
        got_d = float(got_d)
        got_n = float(got_n)
    except Exception:
        return {"mse_within_tol": 0.0, "decomp_wins": 0.0}

    tol = 1e-6
    mse_ok = 1.0 if (abs(got_d - ref_d) <= tol and abs(got_n - ref_n) <= tol) else 0.0
    wins = 1.0 if got_d < got_n else 0.0

    return {"mse_within_tol": mse_ok, "decomp_wins": wins}
