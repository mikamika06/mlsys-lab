import numpy as np


def _oracle(logits, penalty):
    x = np.asarray(logits, dtype=np.float64)
    out = x.copy()
    pos = x > 0
    neg = x < 0
    out[pos] = x[pos] / penalty
    out[neg] = x[neg] * penalty
    return out


def grade(sol, fx) -> dict:
    cases = [
        (np.array([4.0, -3.0, 0.0, 1.5]), 2.0),
        (np.array([-10.0, -0.5, 0.25, 7.0]), 1.25),
        (np.array([0.0, -1.0, 1.0, -2.0, 2.0]), 3.0),
        (np.linspace(-5, 5, 21), 1.7),
    ]

    worst = 0.0
    for logits, penalty in cases:
        try:
            got = sol.apply_repetition_penalty(logits.copy(), penalty)
            got = np.asarray(got, dtype=np.float64)
        except Exception:
            return {"max_abs_err": float("inf")}

        ref = _oracle(logits, penalty)
        if got.shape != ref.shape:
            return {"max_abs_err": float("inf")}
        worst = max(worst, float(np.max(np.abs(got - ref))))

    return {"max_abs_err": worst}
