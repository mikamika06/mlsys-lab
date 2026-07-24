import math

import numpy as np


def _naive(z):
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        ez = np.exp(z)
        return ez / np.sum(ez)


def _lse(z):
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        m = np.max(z)
        ez = np.exp(z - m)
        return ez / np.sum(ez)


def _online(z):
    m = -math.inf
    s = 0.0
    for x in z:
        new_m = max(m, x)
        s = s * math.exp(m - new_m) + math.exp(x - new_m)
        m = new_m
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        return np.exp(z - m) / s


def _overflowed(p):
    return bool(np.any(~np.isfinite(p)))


def _oracle(z):
    z = np.asarray(z, dtype=np.float64)
    return (_overflowed(_naive(z)), _overflowed(_lse(z)), _overflowed(_online(z)))


def _gen_cases(rng):
    cases = []
    # explicit extreme fixed cases (grading hint: scores up to 1e4)
    cases.append(np.array([10000.0, 0.0, -10000.0]))
    cases.append(np.array([1.0, 2.0, 3.0]))
    cases.append(np.array([5000.0, 5000.0, 5000.0]))
    cases.append(np.array([0.0]))
    cases.append(np.array([-10000.0, -10000.0, -10000.0]))
    for _ in range(8):
        n = int(rng.integers(2, 8))
        scale = float(rng.choice([1.0, 10.0, 1000.0, 10000.0]))
        z = rng.standard_normal(n) * scale
        cases.append(z)
    return cases


def grade(sol, fx) -> dict:
    rng = np.random.default_rng(0)
    cases = _gen_cases(rng)

    ok = 1.0
    for z in cases:
        expected = _oracle(z)
        try:
            got = sol.classify_softmax_overflow(z.copy())
            got_norm = tuple(bool(x) for x in got)
        except Exception:
            ok = 0.0
            break
        if got_norm != expected:
            ok = 0.0
            break
    return {"exact_match": ok}
