import numpy as np


def _migration_scales(W: np.ndarray, X: np.ndarray, alpha: float) -> np.ndarray:
    out_c = W.shape[0]
    max_W = np.max(np.abs(W.reshape(out_c, -1)), axis=1)
    max_X = np.max(np.abs(X.reshape(X.shape[0], X.shape[1], -1)), axis=(0, 2))
    return (max_X ** alpha) / (max_W ** (1 - alpha))


def _quantize_int8(t: np.ndarray) -> np.ndarray:
    t = np.asarray(t, dtype=np.float64)
    amax = np.max(np.abs(t))
    if amax < 1e-12:
        return t.copy()
    scale = amax / 127.0
    q = np.clip(np.round(t / scale), -127, 127)
    return q * scale


def _rel_err(orig: np.ndarray, approx: np.ndarray) -> float:
    a = np.asarray(orig, dtype=np.float64).ravel()
    b = np.asarray(approx, dtype=np.float64).ravel()
    return float(np.linalg.norm(b - a) / (np.linalg.norm(a) + 1e-12))


def search_best_alpha(W: np.ndarray, X: np.ndarray, alphas: np.ndarray):
    """
    Grid-search alpha to balance activation vs weight INT8 quantization error.

    For every candidate alpha:
      1. Compute the per-channel migration scale s_j (same formula as
         `compute_migration_scales`).
      2. Migrate: W_mig = W * s (broadcast on channel axis 0),
                  X_mig = X / s (broadcast on channel axis 1).
      3. Quantize both migrated tensors to INT8 with per-tensor symmetric
         round-to-nearest quantization (scale = max(|t|) / 127).
      4. err(alpha) = max( rel_err(X_mig, quant(X_mig)),
                            rel_err(W_mig, quant(W_mig)) ).

    Returns (best_idx, errors) where best_idx = argmin(errors).
    """
    alphas = np.asarray(alphas, dtype=np.float64)
    out_c = W.shape[0]

    w_shape = [1] * W.ndim
    w_shape[0] = out_c
    x_shape = [1] * X.ndim
    x_shape[1] = out_c

    errors = np.zeros(len(alphas), dtype=np.float64)
    for k, alpha in enumerate(alphas):
        s = _migration_scales(W, X, float(alpha))
        W_mig = W * s.reshape(w_shape)
        X_mig = X / s.reshape(x_shape)

        W_q = _quantize_int8(W_mig)
        X_q = _quantize_int8(X_mig)

        w_err = _rel_err(W_mig, W_q)
        x_err = _rel_err(X_mig, X_q)
        errors[k] = max(w_err, x_err)

    best_idx = int(np.argmin(errors))
    return best_idx, errors
