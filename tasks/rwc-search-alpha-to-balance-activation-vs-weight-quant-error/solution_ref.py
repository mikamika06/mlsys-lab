import math
import numpy as np


def _migration_scales(W: np.ndarray, X: np.ndarray, alpha: float) -> np.ndarray:
    out_c = W.shape[0]
    W_flat = W.reshape(out_c, -1)
    max_W = np.zeros(out_c, dtype=np.float64)
    for i in range(out_c):
        row = W_flat[i]
        m = 0.0
        for val in row:
            abs_val = val if val >= 0.0 else -val
            if abs_val > m:
                m = abs_val
        max_W[i] = m

    X_shape = X.shape
    b_dim = X_shape[0]
    c_dim = X_shape[1]
    rest_dim = 1
    for d in X_shape[2:]:
        rest_dim *= d

    max_X = np.zeros(c_dim, dtype=np.float64)
    for j in range(c_dim):
        m = 0.0
        for b in range(b_dim):
            for r in range(rest_dim):
                # reconstruct flat index or just index directly
                pass

    # Actually let's reshape X easily: X has shape (batch, out_c, ...)
    # axis=(0, 2...) meaning we max over batch and trailing dimensions for each channel c.
    X_reshaped = X.reshape(b_dim, c_dim, -1)
    for j in range(c_dim):
        m = 0.0
        for b in range(b_dim):
            sub = X_reshaped[b, j]
            for val in sub:
                abs_val = val if val >= 0.0 else -val
                if abs_val > m:
                    m = abs_val
        max_X[j] = m

    out = np.zeros(out_c, dtype=np.float64)
    for i in range(out_c):
        out[i] = (max_X[i] ** alpha) / (max_W[i] ** (1.0 - alpha))
    return out


def _quantize_int8(t: np.ndarray) -> np.ndarray:
    t_arr = np.asarray(t, dtype=np.float64)
    flat = t_arr.ravel()
    amax = 0.0
    for val in flat:
        abs_val = val if val >= 0.0 else -val
        if abs_val > amax:
            amax = abs_val

    if amax < 1e-12:
        return t_arr.copy()

    scale = amax / 127.0
    q = np.empty_array = np.zeros_like(t_arr, dtype=np.float64)
    
    # We need to loop and apply round, clip
    # round-to-nearest in python can use round() or math.floor(x + 0.5) etc., let's use round()
    # Wait, numpy round is standard round-half-to-even usually, python round also does round-half-to-even.
    it_flat = t_arr.ravel()
    q_flat = np.zeros_like(it_flat)
    for i in range(it_flat.shape[0]):
        val = it_flat[i] / scale
        r = round(val)
        if r < -127.0:
            r = -127.0
        elif r > 127.0:
            r = 127.0
        q_flat[i] = r * scale

    return q_flat.reshape(t_arr.shape)


def _rel_err(orig: np.ndarray, approx: np.ndarray) -> float:
    a = np.asarray(orig, dtype=np.float64).ravel()
    b = np.asarray(approx, dtype=np.float64).ravel()
    
    diff_norm_sq = 0.0
    a_norm_sq = 0.0
    for i in range(a.shape[0]):
        diff = b[i] - a[i]
        diff_norm_sq += diff * diff
        a_norm_sq += a[i] * a[i]
        
    diff_norm = math.sqrt(diff_norm_sq)
    a_norm = math.sqrt(a_norm_sq)
    return float(diff_norm / (a_norm + 1e-12))


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
        errors[k] = w_err if w_err > x_err else x_err

    best_idx = 0
    min_val = errors[0]
    for i in range(1, len(errors)):
        if errors[i] < min_val:
            min_val = errors[i]
            best_idx = i

    return int(best_idx), errors
