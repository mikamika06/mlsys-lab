import numpy as np

from mlsys import scorers

H = 1e-5


def _fns():
    """(f, analytic f') pairs — the analytic derivatives are the oracle."""
    def sig(x):
        return 1.0 / (1.0 + np.exp(-x))

    return [
        (lambda x: x ** 3 - 2.0 * x + 1.0,
         lambda x: 3.0 * x ** 2 - 2.0),
        (lambda x: np.sin(x) * np.exp(-0.3 * x),
         lambda x: np.exp(-0.3 * x) * (np.cos(x) - 0.3 * np.sin(x))),
        (lambda x: np.log(1.0 + x ** 2),
         lambda x: 2.0 * x / (1.0 + x ** 2)),
        (lambda x: np.tanh(2.0 * x),
         lambda x: 2.0 * (1.0 - np.tanh(2.0 * x) ** 2)),
        (sig, lambda x: sig(x) * (1.0 - sig(x))),
        (lambda x: x ** 4 / (1.0 + x ** 2),
         lambda x: (4.0 * x ** 3 * (1.0 + x ** 2) - x ** 4 * 2.0 * x) / (1.0 + x ** 2) ** 2),
    ]


def _points():
    rng = np.random.default_rng(0)
    return [float(v) for v in rng.uniform(-1.5, 1.5, size=12)]


def _ref_central(f, x, h):
    return (f(x + h) - f(x - h)) / (2.0 * h)


def _ref_check(f, g, x, h):
    num = float(_ref_central(f, x, h))
    ana = float(g(x))
    return abs(num - ana) / max(abs(num) + abs(ana), 1e-12)


def _fail():
    return {"rel_err": float("inf"), "check_err": float("inf"), "detect_ok": 0.0}


def grade(sol, fx) -> dict:
    fns = _fns()
    pts = _points()

    got, ana = [], []
    for f, g in fns:
        for x in pts:
            try:
                d = float(sol.central_diff(f, x, H))
            except Exception:
                return _fail()
            if not np.isfinite(d):
                return _fail()
            got.append(d)
            ana.append(float(g(x)))

    rel = scorers.rel_err(np.asarray(ana), np.asarray(got))

    # grad_check must reproduce the oracle's relative-difference formula
    worst_check = 0.0
    detect_ok = 1.0
    for f, g in fns:
        for x in pts:
            try:
                r = float(sol.grad_check(f, g, x, H))
            except Exception:
                return {"rel_err": float(rel), "check_err": float("inf"), "detect_ok": 0.0}
            ref = _ref_check(f, g, x, H)
            worst_check = max(worst_check, abs(r - ref))
            if r > 1e-6:
                detect_ok = 0.0

    # a deliberately wrong analytic gradient must be flagged
    for f, g in fns:
        bad = (lambda gg: (lambda x: 1.5 * gg(x) + 0.1))(g)
        for x in pts:
            try:
                r = float(sol.grad_check(f, bad, x, H))
            except Exception:
                return {"rel_err": float(rel), "check_err": float("inf"), "detect_ok": 0.0}
            ref = _ref_check(f, bad, x, H)
            worst_check = max(worst_check, abs(r - ref))
            if ref > 1e-3 and r <= 1e-3:
                detect_ok = 0.0

    return {
        "rel_err": float(rel),
        "check_err": float(worst_check),
        "detect_ok": float(detect_ok),
    }
