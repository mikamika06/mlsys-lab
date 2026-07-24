import numpy as np

from mlsys import scorers

MAGIC = np.uint32(0x5F3759DF)


def _oracle_raw(x32):
    """The bit hack itself, done with a real uint32 view — the oracle."""
    i = x32.view(np.uint32)
    j = MAGIC - (i >> np.uint32(1))
    return j.view(np.float32)


def _inputs():
    rng = np.random.default_rng(0)
    a = (10.0 ** rng.uniform(-6.0, 6.0, size=4096)).astype(np.float32)
    b = (2.0 ** np.arange(-40, 40, dtype=np.float64)).astype(np.float32)
    c = np.array([1.0, 2.0, 4.0, 0.25, 3.0, 1e-3, 1e3], dtype=np.float32)
    return np.concatenate([a, b, c]).astype(np.float32)


def _rel(ref64, got32):
    got = np.asarray(got32, dtype=np.float64)
    return np.abs(got - ref64) / np.abs(ref64)


def _fail():
    return {
        "raw_bit_exact": 0.0,
        "newton_max_rel_err": float("inf"),
        "newton_rel_err": float("inf"),
        "newton_dtype_ok": 0.0,
        "raw_max_rel_err": float("nan"),
    }


def grade(sol, fx) -> dict:
    x = _inputs()
    exact = 1.0 / np.sqrt(x.astype(np.float64))  # float64 oracle for 1/sqrt(x)
    ref_raw = _oracle_raw(x.copy())

    try:
        got_raw = sol.rsqrt_raw(x.copy())
    except Exception:
        return _fail()
    try:
        got_raw = np.asarray(got_raw)
    except Exception:
        return _fail()

    if got_raw.shape != x.shape:
        bit_exact = 0.0
        raw_rel = float("nan")
    else:
        bit_exact = float(scorers.byte_exact_fraction(ref_raw, got_raw))
        if np.all(np.isfinite(np.asarray(got_raw, dtype=np.float64))):
            raw_rel = float(np.max(_rel(exact, got_raw)))
        else:
            raw_rel = float("nan")

    try:
        got_n = np.asarray(sol.rsqrt_newton(x.copy()))
    except Exception:
        return {**_fail(), "raw_bit_exact": bit_exact, "raw_max_rel_err": raw_rel}

    if got_n.shape != x.shape or not np.all(np.isfinite(np.asarray(got_n, dtype=np.float64))):
        return {**_fail(), "raw_bit_exact": bit_exact, "raw_max_rel_err": raw_rel}

    n_max = float(np.max(_rel(exact, got_n)))
    n_l2 = float(scorers.rel_err(exact, np.asarray(got_n, dtype=np.float64)))
    dtype_ok = 1.0 if np.asarray(got_n).dtype == np.float32 else 0.0

    return {
        "raw_bit_exact": bit_exact,
        "newton_max_rel_err": n_max,
        "newton_rel_err": n_l2,
        "newton_dtype_ok": dtype_ok,
        "raw_max_rel_err": raw_rel,
    }
