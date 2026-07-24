import numpy as np


def _create_dynamic_map(max_exponent_bits=7, total_bits=8):
    data = []
    non_sign_bits = total_bits - 1
    for i in range(max_exponent_bits):
        fraction_items = int(2 ** i + 1)
        boundaries = np.linspace(0.1, 1, fraction_items)
        means = (boundaries[:-1] + boundaries[1:]) / 2.0
        v = (10.0 ** (i - (max_exponent_bits - 1))) * means
        data += list(v)
        data += list(-v)
    data.append(0.0)
    data.append(1.0)
    assert len(data) == 256
    return np.array(sorted(data), dtype=np.float64)


def _nearest_dequant(scaled, codes):
    idx = np.searchsorted(codes, scaled)
    idx = np.clip(idx, 1, len(codes) - 1)
    left = codes[idx - 1]
    right = codes[idx]
    choose_left = (scaled - left) < (right - scaled)
    return np.where(choose_left, left, right)


def _oracle_mses(x):
    x = np.asarray(x, dtype=np.float64)
    absmax = np.max(np.abs(x))

    codes = _create_dynamic_map()
    scaled = x / absmax
    deq_dyn = _nearest_dequant(scaled, codes) * absmax
    mse_dyn = float(np.mean((deq_dyn - x) ** 2))

    scale_lin = absmax / 127.0
    q = np.clip(np.round(x / scale_lin), -127, 127)
    deq_lin = q * scale_lin
    mse_lin = float(np.mean((deq_lin - x) ** 2))

    return mse_dyn, mse_lin


def grade(sol, fx) -> dict:
    rng = np.random.default_rng(0)
    mag = rng.lognormal(mean=0.0, sigma=1.2, size=4096)
    sign = rng.choice([-1.0, 1.0], size=4096)
    x = (mag * sign).astype(np.float64)

    ref_dyn, ref_lin = _oracle_mses(x)

    try:
        got_dyn, got_lin = sol.dynamic_vs_linear_int8_mse(x)
        got_dyn = float(got_dyn)
        got_lin = float(got_lin)
    except Exception:
        return {"rel_err": float("inf"), "dynamic_wins": 0.0}

    denom = max(abs(ref_dyn), abs(ref_lin), 1e-12)
    rel_err = max(abs(got_dyn - ref_dyn), abs(got_lin - ref_lin)) / denom
    dynamic_wins = 1.0 if got_dyn < got_lin else 0.0

    return {"rel_err": float(rel_err), "dynamic_wins": dynamic_wins}
