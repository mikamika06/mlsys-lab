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
        ([4.0, -3.0, 0.0, 1.5], 2.0),
        ([-10.0, -0.5, 0.25, 7.0], 1.25),
        ([0.0, -1.0, 1.0, -2.0, 2.0], 3.0),
        (list(np.linspace(-5, 5, 21)), 1.7),
    ]

    worst = 0.0
    for logits, penalty in cases:
        try:
            got = sol.apply_repetition_penalty(list(logits), penalty)
            got_arr = np.asarray(got, dtype=np.float64)
        except Exception:
            return {"max_abs_err": float("inf")}

        ref = _oracle(logits, penalty)
        if got_arr.shape != ref.shape:
            return {"max_abs_err": float("inf")}
        worst = max(worst, float(np.max(np.abs(got_arr - ref))))

    return {"max_abs_err": worst}
