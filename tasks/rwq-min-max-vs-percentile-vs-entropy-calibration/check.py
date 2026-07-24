import numpy as np


def _oracle_entropy_threshold(x_abs: np.ndarray, num_bits: int) -> float:
    num_quant_bins = 2 ** (num_bits - 1)
    num_bins = 16 * num_quant_bins

    x_max = float(x_abs.max())
    if x_max <= 0.0:
        return 1e-12

    hist, edges = np.histogram(x_abs, bins=num_bins, range=(0.0, x_max))
    bin_width = edges[1] - edges[0]
    hist = hist.astype(np.float64)

    eps = 1e-8
    best_kl = np.inf
    best_i = num_quant_bins

    for i in range(num_quant_bins, num_bins + 1, num_quant_bins):
        P = hist[:i].copy()
        P[-1] += hist[i:].sum()
        P_sum = P.sum()
        if P_sum <= 0:
            continue

        group_size = i // num_quant_bins
        Q = np.zeros(i, dtype=np.float64)
        for g in range(num_quant_bins):
            g0, g1 = g * group_size, (g + 1) * group_size
            group = P[g0:g1]
            group_sum = group.sum()
            nonzero = group > 0
            n_nonzero = int(nonzero.sum())
            if n_nonzero > 0 and group_sum > 0:
                Q[g0:g1][nonzero] = group_sum / n_nonzero

        Q_sum = Q.sum()
        if Q_sum <= 0:
            continue

        Pn = P / P_sum
        Pn = np.where(Pn == 0, eps, Pn)
        Pn = Pn / Pn.sum()

        Qn = Q / Q_sum
        Qn = np.where(Qn == 0, eps, Qn)
        Qn = Qn / Qn.sum()

        kl = float(np.sum(Pn * np.log(Pn / Qn)))
        if kl < best_kl:
            best_kl = kl
            best_i = i

    return best_i * bin_width


def _oracle_quantize_mse(x: np.ndarray, threshold: float, num_bits: int) -> float:
    levels = 2 ** (num_bits - 1) - 1
    threshold = max(threshold, 1e-12)
    scale = threshold / levels
    q = np.clip(np.round(x / scale), -levels, levels)
    x_hat = q * scale
    return float(np.mean((x - x_hat) ** 2))


def _oracle_calibrate(X: np.ndarray, num_bits: int) -> dict:
    x = np.asarray(X, dtype=np.float64).ravel()
    x_abs = np.abs(x)
    t_minmax = float(x_abs.max())
    t_percentile = float(np.percentile(x_abs, 99.99))
    t_entropy = _oracle_entropy_threshold(x_abs, num_bits)
    return {
        "minmax": _oracle_quantize_mse(x, t_minmax, num_bits),
        "percentile": _oracle_quantize_mse(x, t_percentile, num_bits),
        "entropy": _oracle_quantize_mse(x, t_entropy, num_bits),
    }


def _make_activation(rng, n, df):
    return rng.standard_t(df, size=n)


def grade(sol, fx) -> dict:
    num_bits = 8
    cases = [(0, 3000, 1.5), (1, 5000, 2.0), (2, 8000, 3.0)]
    keys = ("minmax", "percentile", "entropy")

    sq_err_sum = 0.0
    n_vals = 0
    argmin_ok = 1.0

    for seed, n, df in cases:
        rng = np.random.default_rng(seed)
        X = _make_activation(rng, n, df)
        ref = _oracle_calibrate(X, num_bits)

        try:
            got = sol.calibrate_and_score(X.copy(), num_bits)
        except Exception:
            return {"mse": float("inf"), "argmin_match": 0.0}

        if not isinstance(got, dict) or not all(k in got for k in keys):
            return {"mse": float("inf"), "argmin_match": 0.0}

        ref_vec = np.array([ref[k] for k in keys], dtype=np.float64)
        try:
            got_vec = np.array([float(got[k]) for k in keys], dtype=np.float64)
        except Exception:
            return {"mse": float("inf"), "argmin_match": 0.0}

        if not np.all(np.isfinite(got_vec)):
            return {"mse": float("inf"), "argmin_match": 0.0}

        sq_err_sum += float(np.sum((got_vec - ref_vec) ** 2))
        n_vals += len(keys)

        if int(np.argmin(got_vec)) != int(np.argmin(ref_vec)):
            argmin_ok = 0.0

    mse = sq_err_sum / n_vals if n_vals else float("inf")
    return {"mse": mse, "argmin_match": argmin_ok}
