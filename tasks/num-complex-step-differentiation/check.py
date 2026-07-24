import numpy as np

def grade(sol, fx) -> dict:
    """Grade complex-step differentiation against analytic oracles.

    Returns {"rel_err": <max relative error across test cases>}.
    """
    # (function, evaluation point, analytic derivative)
    cases = [
        (np.sin, 1.0, np.cos(1.0)),
        (np.cos, 2.0, -np.sin(2.0)),
        (np.exp, 3.0, np.exp(3.0)),
        (lambda x: x ** 2, 5.0, 10.0),
        (lambda x: x ** 3, 2.0, 12.0),
        (np.log, 1.5, 1.0 / 1.5),
        (lambda x: np.sin(x) * np.exp(x), 1.0,
         (np.cos(1.0) + np.sin(1.0)) * np.exp(1.0)),
        (lambda x: x ** 4 - 3 * x ** 2 + 7, 2.0, 4 * 2 ** 3 - 6 * 2),
    ]

    max_err = 0.0
    for f, x, analytic in cases:
        try:
            approx = float(sol.complex_step_diff(f, x))
        except Exception:
            return {"rel_err": 1.0}
        denom = max(abs(analytic), 1e-15)
        err = abs(approx - analytic) / denom
        if err > max_err:
            max_err = err

    return {"rel_err": max_err}
