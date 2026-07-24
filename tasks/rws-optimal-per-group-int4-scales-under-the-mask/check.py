import numpy as np


def _naive_group_dequant(seg, qmax):
    amax = float(np.max(np.abs(seg)))
    scale = amax / qmax if amax > 0 else 1.0
    dequant = scale * np.clip(np.round(seg / scale), -qmax, qmax)
    return scale, dequant


def _oracle(W, M, X, group_size, bits, alphas):
    W = np.asarray(W, dtype=np.float64)
    M = np.asarray(M, dtype=np.float64)
    X = np.asarray(X, dtype=np.float64)
    qmax = (1 << (bits - 1)) - 1
    Wm = W * M
    O, I = Wm.shape
    n_groups = I // group_size

    what = np.zeros_like(Wm)
    for o in range(O):
        for g in range(n_groups):
            sl = slice(g * group_size, (g + 1) * group_size)
            _s, dq = _naive_group_dequant(Wm[o, sl], qmax)
            what[o, sl] = dq
    what_naive = what.copy()

    group_scales = np.zeros((O, n_groups), dtype=np.float64)
    for o in range(O):
        row_target = Wm[o]
        row_what = what[o].copy()
        for g in range(n_groups):
            sl = slice(g * group_size, (g + 1) * group_size)
            seg = row_target[sl]
            amax = float(np.max(np.abs(seg)))
            best_err = np.inf
            best_scale = None
            best_dq = None
            for alpha in alphas:
                scale = (alpha * amax / qmax) if amax > 0 else 1.0
                dq = scale * np.clip(np.round(seg / scale), -qmax, qmax)
                trial = row_what.copy()
                trial[sl] = dq
                err = float(np.sum((X @ (row_target - trial)) ** 2))
                if err < best_err:
                    best_err = err
                    best_scale = scale
                    best_dq = dq
            row_what[sl] = best_dq
            group_scales[o, g] = best_scale
        what[o] = row_what

    Y = X @ Wm.T
    Yhat = X @ what.T
    mse = float(np.mean((Y - Yhat) ** 2))

    Y_naive = X @ what_naive.T
    naive_mse = float(np.mean((Y - Y_naive) ** 2))

    return group_scales, mse, naive_mse


def _build_cases():
    cases = []
    alphas = np.linspace(0.6, 1.4, 9)
    for seed, O, I, group_size, bits, prune_frac in [
        (0, 6, 24, 8, 4, 0.0),
        (1, 5, 16, 8, 4, 0.3),
        (2, 4, 12, 4, 3, 0.2),
    ]:
        rng = np.random.default_rng(seed)
        W = rng.standard_normal((O, I))
        M = (rng.uniform(size=(O, I)) >= prune_frac).astype(np.float64)
        X = rng.standard_normal((12, I))
        cases.append((W, M, X, group_size, bits, alphas))
    return cases


def grade(sol, fx) -> dict:
    worst_mse_rel = 0.0
    worst_margin = float("inf")

    for W, M, X, group_size, bits, alphas in _build_cases():
        scales_ref, mse_ref, naive_mse_ref = _oracle(W, M, X, group_size, bits, alphas)

        try:
            out = sol.optimal_group_scales_under_mask(W.copy(), M.copy(), X.copy(), group_size,
                                                        bits=bits, alphas=alphas.copy())
            scales_got, mse_got = out
            scales_got = np.asarray(scales_got, dtype=np.float64)
            mse_got = float(mse_got)
        except Exception:
            return {"mse_rel_err": float("inf"), "mse_vs_naive_margin": float("-inf")}

        if scales_got.shape != scales_ref.shape or not np.isfinite(mse_got):
            return {"mse_rel_err": float("inf"), "mse_vs_naive_margin": float("-inf")}

        rel = abs(mse_got - mse_ref) / (abs(mse_ref) + 1e-12)
        worst_mse_rel = max(worst_mse_rel, rel)
        worst_margin = min(worst_margin, naive_mse_ref - mse_got)

    return {"mse_rel_err": worst_mse_rel, "mse_vs_naive_margin": worst_margin}
