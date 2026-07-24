import numpy as np
import ml_dtypes

_DTYPES = {"e4m3": ml_dtypes.float8_e4m3fn, "e5m2": ml_dtypes.float8_e5m2}


def _oracle(x, fmt):
    x = np.asarray(x, dtype=np.float32)
    dt = _DTYPES[fmt]
    return x.astype(dt).astype(np.float32)


def _cases():
    rng = np.random.default_rng(0)
    cases = []
    for fmt in ("e4m3", "e5m2"):
        cases.append((rng.uniform(-300.0, 300.0, size=200).astype(np.float32), fmt))
        cases.append((rng.standard_normal(150).astype(np.float32) * 5.0, fmt))
        cases.append((rng.uniform(-0.02, 0.02, size=100).astype(np.float32), fmt))
        small = np.zeros(5, dtype=np.float32)
        cases.append((small, fmt))
    return cases


def grade(sol, fx) -> dict:
    worst = 0.0
    for x, fmt in _cases():
        ref = _oracle(x, fmt)
        try:
            got = np.asarray(sol.fp8_round_trip(x, fmt), dtype=np.float32)
        except Exception:
            return {"max_abs_err": float("inf")}
        if got.shape != ref.shape:
            return {"max_abs_err": float("inf")}
        err = float(np.max(np.abs(got - ref)))
        worst = max(worst, err)
    return {"max_abs_err": worst}
