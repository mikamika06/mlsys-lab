from decimal import Decimal, getcontext

import numpy as np

from mlsys import scorers

_FAIL = {"max_abs_err": float("inf"), "overflow_free": 0.0, "range_ok": 0.0}


def _oracle(xs: np.ndarray) -> np.ndarray:
    """sigma(x) from CPython's arbitrary-precision decimal module (50 digits)."""
    ctx = getcontext()
    old = ctx.prec
    ctx.prec = 50
    try:
        out = np.empty(xs.size, dtype=np.float64)
        for i, v in enumerate(xs.ravel()):
            d = Decimal(float(v))
            out[i] = float(Decimal(1) / (Decimal(1) + (-d).exp()))
    finally:
        ctx.prec = old
    return out.reshape(xs.shape)


def _points() -> np.ndarray:
    rng = np.random.default_rng(0)
    moderate = rng.uniform(-30.0, 30.0, 200)
    tiny = rng.uniform(-1e-8, 1e-8, 10)
    extreme = np.array([
        -1e4, -1000.0, -800.0, -745.0, -709.0, -100.0, -50.0,
        50.0, 100.0, 709.0, 745.0, 800.0, 1000.0, 1e4,
    ])
    return np.concatenate([moderate, tiny, extreme, np.array([0.0])]).astype(np.float64)


def grade(sol, fx) -> dict:
    x = _points()

    try:
        got = np.asarray(sol.stable_sigmoid(x.copy()), dtype=np.float64)
    except Exception:
        return dict(_FAIL)
    if got.shape != x.shape:
        return dict(_FAIL)

    ref = _oracle(x)
    if not np.all(np.isfinite(got)):
        return dict(_FAIL)
    err = scorers.max_abs_err(ref, got)

    # --- overflow_free: must survive strict floating-point error state ---
    overflow_free = 1.0
    hard = np.array([-1e4, -1000.0, -710.0, 0.0, 710.0, 1000.0, 1e4], dtype=np.float64)
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            probe_out = np.asarray(sol.stable_sigmoid(hard.copy()), dtype=np.float64)
        if not np.all(np.isfinite(probe_out)):
            overflow_free = 0.0
    except Exception:
        overflow_free = 0.0

    # --- range_ok: [0,1], finite, and the reflection identity ---
    range_ok = 1.0
    if np.any(got < 0.0) or np.any(got > 1.0):
        range_ok = 0.0
    try:
        neg = np.asarray(sol.stable_sigmoid(-x.copy()), dtype=np.float64)
        if neg.shape != x.shape or not np.all(np.isfinite(neg)):
            range_ok = 0.0
        elif float(np.max(np.abs((neg + got) - 1.0))) > 1e-12:
            range_ok = 0.0
    except Exception:
        range_ok = 0.0

    return {
        "max_abs_err": float(err),
        "overflow_free": float(overflow_free),
        "range_ok": float(range_ok),
    }
