import numpy as np


def _migration_scales(W, X, alpha):
    out_c = W.shape[0]
    max_W = np.max(np.abs(W.reshape(out_c, -1)), axis=1)
    max_X = np.max(np.abs(X.reshape(X.shape[0], X.shape[1], -1)), axis=(0, 2))
    return (max_X ** alpha) / (max_W ** (1 - alpha))


def _quantize_int8(t):
    t = np.asarray(t, dtype=np.float64)
    amax = np.max(np.abs(t))
    if amax < 1e-12:
        return t.copy()
    scale = amax / 127.0
    q = np.clip(np.round(t / scale), -127, 127)
    return q * scale


def _rel_err(orig, approx):
    a = np.asarray(orig, dtype=np.float64).ravel()
    b = np.asarray(approx, dtype=np.float64).ravel()
    return float(np.linalg.norm(b - a) / (np.linalg.norm(a) + 1e-12))


def _oracle(W, X, alphas):
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
        errors[k] = max(_rel_err(W_mig, W_q), _rel_err(X_mig, X_q))
    return int(np.argmin(errors)), errors


def _cases():
    rng = np.random.default_rng(0)
    cases = []

    # Conv-like tensors, moderate channel imbalance.
    out_c, in_c, kH, kW = 6, 4, 3, 3
    batch, h, w = 5, 6, 6
    W = rng.standard_normal((out_c, in_c, kH, kW))
    # give some weight channels a much larger dynamic range
    boost_w = rng.uniform(1.0, 20.0, size=out_c)
    W = W * boost_w[:, None, None, None]
    X = rng.standard_normal((batch, out_c, h, w))
    boost_x = rng.uniform(1.0, 20.0, size=out_c)
    X = X * boost_x[None, :, None, None]
    alphas = np.linspace(0.0, 1.0, 11)
    cases.append((W, X, alphas))

    # Linear-like tensors (2-D), activations much larger than weights.
    out_c2, in_c2 = 8, 5
    batch2 = 20
    W2 = rng.standard_normal((out_c2, in_c2)) * 0.05
    X2 = rng.standard_normal((batch2, out_c2)) * 50.0
    alphas2 = np.linspace(0.0, 1.0, 9)
    cases.append((W2, X2, alphas2))

    # Weights much larger than activations.
    out_c3, in_c3 = 5, 3
    batch3 = 15
    W3 = rng.standard_normal((out_c3, in_c3)) * 40.0
    X3 = rng.standard_normal((batch3, out_c3)) * 0.1
    alphas3 = np.linspace(0.0, 1.0, 13)
    cases.append((W3, X3, alphas3))

    return cases


def grade(sol, fx) -> dict:
    idx_scores = []
    curve_errs = []

    for W, X, alphas in _cases():
        ref_idx, ref_errors = _oracle(W, X, alphas)
        try:
            got = sol.search_best_alpha(W.copy(), X.copy(), alphas.copy())
            got_idx, got_errors = got
            got_idx = int(got_idx)
            got_errors = np.asarray(got_errors, dtype=np.float64)
        except Exception:
            idx_scores.append(0.0)
            curve_errs.append(1.0)
            continue

        # Tolerant argmin check: accept the chosen index if it achieves an
        # error within 1e-6 of the true minimum (robust to ties on the grid).
        if 0 <= got_idx < len(alphas) and ref_errors[got_idx] <= ref_errors[ref_idx] + 1e-6:
            idx_scores.append(1.0)
        else:
            idx_scores.append(0.0)

        if got_errors.shape == ref_errors.shape:
            curve_errs.append(_rel_err(ref_errors, got_errors))
        else:
            curve_errs.append(1.0)

    return {
        "idx_ok": float(min(idx_scores)) if idx_scores else 0.0,
        "curve_rel_err": float(max(curve_errs)) if curve_errs else 1.0,
    }
