import numpy as np


def _central_oracle_derivative(f, x, hs):
    h = float(np.min(hs)) * 0.01
    return (f(x + h) - f(x - h)) / (2.0 * h)


def _ref_errors(f, x, hs):
    true_derivative = _central_oracle_derivative(f, x, hs)
    fx = f(x)
    forward = []
    central = []
    for h in hs:
        fd = (f(x + h) - fx) / h
        cd = (f(x + h) - f(x - h)) / (2.0 * h)
        forward.append(abs(fd - true_derivative))
        central.append(abs(cd - true_derivative))
    return np.asarray(forward, dtype=np.float64), np.asarray(central, dtype=np.float64)


def _pack(a, b):
    return np.concatenate([np.asarray(a, dtype=np.float64),
                           np.asarray(b, dtype=np.float64)])


def grade(sol, fx) -> dict:
    cases = [
        (lambda x: np.sin(x), 0.7, np.logspace(-1, -5, 5)),
        (lambda x: np.exp(x), -0.4, np.logspace(-1, -5, 5)),
        (lambda x: x**3 + 2.0*x, 1.2, np.logspace(-1, -6, 6)),
    ]

    refs = []
    got = []
    for f, x, hs in cases:
        try:
            a, b = sol.finite_difference_error_orders(f, x, hs)
        except Exception:
            return {"rel_err": 1.0}

        ra, rb = _ref_errors(f, x, hs)
        refs.append(_pack(ra, rb))
        got.append(_pack(a, b))

        for curve in (a, b):
            curve = np.asarray(curve, dtype=np.float64)
            if curve.shape != hs.shape:
                return {"rel_err": 1.0}

    ref = np.concatenate(refs)
    cand = np.concatenate(got)
    err = float(np.linalg.norm(cand - ref) /
                (np.linalg.norm(ref) + 1e-12))

    return {"rel_err": err}
