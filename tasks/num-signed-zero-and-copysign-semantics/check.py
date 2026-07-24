import numpy as np


def _oracle():
    with np.errstate(divide="ignore", invalid="ignore"):
        values = np.array([
            1.0 / np.float64(-0.0),
            np.float64(-0.0) + np.float64(0.0),
            np.float64(0.0) + np.float64(-0.0),
            np.copysign(np.float64(0.0), np.float64(-1.0)),
            np.copysign(np.float64(-0.0), np.float64(1.0)),
            np.copysign(np.float64(5.0), np.float64(-0.0)),
        ], dtype=np.float64)
    return [int(x) for x in np.signbit(values)]


def grade(sol, fx) -> dict:
    expected = _oracle()
    try:
        got = sol.signed_zero_profile()
        ok = 1.0 if list(got) == expected else 0.0
    except Exception:
        ok = 0.0
    return {"exact_match": ok}
