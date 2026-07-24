import numpy as np

# The fixed NF4 codebook (bitsandbytes QLoRA), 16 quantile-derived levels in [-1, 1].
_NF4 = np.array([
    -1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
    -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0,
    0.07958029955625534, 0.16093020141124725, 0.24611230194568634,
    0.33791524171829224, 0.44070982933044434, 0.5626170039176941,
    0.7229568362236023, 1.0,
], dtype=np.float64)


def _nf4_mse(w: np.ndarray) -> float:
    absmax = float(np.max(np.abs(w))) or 1.0
    wn = w / absmax
    d = np.abs(wn[:, None] - _NF4[None, :])
    idx = np.argmin(d, axis=1)
    deq = _NF4[idx] * absmax
    return float(np.mean((w - deq) ** 2))


def _int4_mse(w: np.ndarray) -> float:
    absmax = float(np.max(np.abs(w))) or 1.0
    scale = absmax / 7.0
    codes = np.clip(np.round(w / scale), -8, 7)
    deq = codes * scale
    return float(np.mean((w - deq) ** 2))


def grade(sol, fx) -> dict:
    rng = np.random.default_rng(0)
    worst_abs = 0.0
    ordering_ok = 1.0
    n = 256  # large enough that NF4 < INT4 holds reliably for ~Gaussian data

    for _ in range(5):
        w = rng.normal(scale=float(rng.uniform(0.5, 2.0)), size=n)

        exp_nf4 = _nf4_mse(w)
        exp_int4 = _int4_mse(w)
        assert exp_nf4 < exp_int4, "fixture must favor NF4 (sanity check on the oracle itself)"

        try:
            got = sol.nf4_vs_int4_mse(w.copy())
            got_nf4, got_int4 = float(got[0]), float(got[1])
        except Exception:
            return {"max_abs_err": float("inf"), "ordering_ok": 0.0}

        worst_abs = max(worst_abs, abs(got_nf4 - exp_nf4), abs(got_int4 - exp_int4))
        if not (got_nf4 < got_int4):
            ordering_ok = 0.0

    return {"max_abs_err": worst_abs, "ordering_ok": ordering_ok}
