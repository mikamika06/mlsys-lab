import math


def _oracle(f, df, x):
    hs = [10.0 ** k for k in range(-16, 0)]
    true = df(x)
    best_h = None
    best_err = float("inf")
    for h in hs:
        estimate = (f(x + h) - f(x - h)) / (2.0 * h)
        err = abs(estimate - true)
        if err < best_err:
            best_err = err
            best_h = h
    return best_h


def grade(sol, fx) -> dict:
    cases = [
        (math.sin, math.cos, 1.0),
        (math.exp, math.exp, 0.5),
        (math.log, lambda x: 1.0 / x, 2.0),
        (lambda x: math.sin(x) * math.cos(x),
         lambda x: math.cos(x) * math.cos(x) - math.sin(x) * math.sin(x),
         0.7),
    ]

    errs = []
    for f, df, x in cases:
        try:
            got = float(sol.optimal_fd_step(f, df, x))
        except Exception:
            return {"rel_err": 1.0}
        ref = _oracle(f, df, x)
        errs.append(abs(got - ref) / (abs(ref) + 1e-12))

    return {"rel_err": max(errs)}
