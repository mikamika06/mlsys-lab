import numpy as np

E4M3_MAX = 448.0


def _e4m3_grid() -> np.ndarray:
    bias = 7
    vals = set()
    for sign in (1.0, -1.0):
        for e in range(16):
            for m in range(8):
                if e == 15 and m == 7:
                    continue  # NaN
                if e == 0:
                    v = (m / 8.0) * (2.0 ** (1 - bias))
                else:
                    v = (1.0 + m / 8.0) * (2.0 ** (e - bias))
                vals.add(sign * v)
    return np.array(sorted(vals), dtype=np.float64)


_GRID = _e4m3_grid()


def _round_to_e4m3(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    flat = x.reshape(-1)
    idx = np.argmin(np.abs(flat[:, None] - _GRID[None, :]), axis=1)
    return _GRID[idx].reshape(x.shape)


def _oracle(W: np.ndarray):
    W = np.asarray(W, dtype=np.float64)
    amax = float(np.max(np.abs(W)))
    scale = amax / E4M3_MAX if amax > 0 else 1.0
    W_scaled = W / scale
    codes = _round_to_e4m3(W_scaled)
    W_hat = codes * scale
    return scale, codes, W_hat


def _build_cases():
    cases = []
    for seed, shape, mag in [(0, (256,), 5.0), (1, (16, 16), 300.0), (2, (64,), 0.02)]:
        rng = np.random.default_rng(seed)
        W = rng.standard_normal(shape) * mag
        W[0] = 0.0  # exercise the exact-zero path
        cases.append(W)
    return cases


def grade(sol, fx) -> dict:
    worst_scale_err = 0.0
    worst_dequant_err = 0.0

    for W in _build_cases():
        scale_ref, codes_ref, W_hat_ref = _oracle(W)

        try:
            got = sol.qfloat8_weight_quant(W.copy())
            scale_got, codes_got, W_hat_got = got
            scale_got = float(scale_got)
            codes_got = np.asarray(codes_got, dtype=np.float64)
            W_hat_got = np.asarray(W_hat_got, dtype=np.float64)
        except Exception:
            return {"scale_abs_err": float("inf"), "dequant_max_abs_err": float("inf")}

        if codes_got.shape != codes_ref.shape or W_hat_got.shape != W_hat_ref.shape:
            return {"scale_abs_err": float("inf"), "dequant_max_abs_err": float("inf")}
        if not (np.all(np.isfinite(codes_got)) and np.all(np.isfinite(W_hat_got))):
            return {"scale_abs_err": float("inf"), "dequant_max_abs_err": float("inf")}

        worst_scale_err = max(worst_scale_err, abs(scale_got - scale_ref))
        worst_dequant_err = max(worst_dequant_err, float(np.max(np.abs(W_hat_got - W_hat_ref))))

    return {"scale_abs_err": worst_scale_err, "dequant_max_abs_err": worst_dequant_err}
