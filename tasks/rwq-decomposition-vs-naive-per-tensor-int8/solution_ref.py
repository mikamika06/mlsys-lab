import numpy as np

def compare_quantization(X, threshold=6.0):
    """Compare LLM.int8() decomposition vs naive per-tensor int8."""
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
    return (mse_decomp, mse_naive)
