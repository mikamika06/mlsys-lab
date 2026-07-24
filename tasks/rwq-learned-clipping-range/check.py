import numpy as np

from mlsys import scorers


def _mse_at_alpha(group: np.ndarray, alpha: float, qmax: int) -> float:
    amax = float(np.max(np.abs(group)))
    c = alpha * amax
    if c <= 1e-12:
        c = 1e-12
    scale = c / qmax
    codes = np.clip(np.round(group / scale), -qmax, qmax)
    ghat = codes * scale
    return float(np.mean((ghat - group) ** 2))


def _optimize_group(group, qmax, n_steps, lr, eps):
    alpha = 1.0
    for _ in range(n_steps):
        f_plus = _mse_at_alpha(group, alpha + eps, qmax)
        f_minus = _mse_at_alpha(group, alpha - eps, qmax)
        grad = (f_plus - f_minus) / (2.0 * eps)
        alpha = alpha - lr * float(np.sign(grad))
        alpha = min(max(alpha, 0.2), 1.5)
    mse = _mse_at_alpha(group, alpha, qmax)
    return alpha, mse


def _oracle(w, group_size, bits, n_steps, lr, eps):
    w = np.asarray(w, dtype=np.float64)
    n_groups = w.shape[0] // group_size
    qmax = (1 << (bits - 1)) - 1
    alphas = np.zeros(n_groups, dtype=np.float64)
    mses = np.zeros(n_groups, dtype=np.float64)
    for g in range(n_groups):
        seg = w[g * group_size:(g + 1) * group_size]
        a, m = _optimize_group(seg, qmax, n_steps, lr, eps)
        alphas[g] = a
        mses[g] = m
    return alphas, mses


def _build_cases():
    cases = []
    for seed, n, group_size, bits in [(0, 256, 32, 4), (1, 128, 16, 3), (2, 96, 24, 4)]:
        rng = np.random.default_rng(seed)
        w = rng.standard_normal(n)
        # sprinkle a few outliers so the clip trade-off is real
        outlier_idx = rng.choice(n, size=max(1, n // group_size), replace=False)
        w[outlier_idx] *= rng.uniform(5.0, 12.0, size=outlier_idx.shape)
        cases.append((w, group_size, bits, 25, 0.05, 1e-3))
    return cases


def grade(sol, fx) -> dict:
    worst_alpha_num = 0.0
    worst_alpha_den = 0.0
    worst_mse_err = 0.0

    all_alpha_ref = []
    all_alpha_got = []

    for w, group_size, bits, n_steps, lr, eps in _build_cases():
        alphas_ref, mses_ref = _oracle(w, group_size, bits, n_steps, lr, eps)

        try:
            out = sol.learned_clip_range(w.copy(), group_size, bits, n_steps=n_steps, lr=lr, eps=eps)
            alphas_got, mses_got = out
            alphas_got = np.asarray(alphas_got, dtype=np.float64)
            mses_got = np.asarray(mses_got, dtype=np.float64)
        except Exception:
            return {"alpha_rel_err": float("inf"), "mse_abs_err": float("inf")}

        if alphas_got.shape != alphas_ref.shape or mses_got.shape != mses_ref.shape:
            return {"alpha_rel_err": float("inf"), "mse_abs_err": float("inf")}
        if not (np.all(np.isfinite(alphas_got)) and np.all(np.isfinite(mses_got))):
            return {"alpha_rel_err": float("inf"), "mse_abs_err": float("inf")}

        all_alpha_ref.append(alphas_ref)
        all_alpha_got.append(alphas_got)
        worst_mse_err = max(worst_mse_err, scorers.max_abs_err(mses_ref, mses_got))

    alpha_rel_err = scorers.rel_err(np.concatenate(all_alpha_ref), np.concatenate(all_alpha_got))

    return {"alpha_rel_err": alpha_rel_err, "mse_abs_err": worst_mse_err}
