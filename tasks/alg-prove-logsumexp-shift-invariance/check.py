import numpy as np

def grade(sol, fx) -> dict:
    rng = np.random.default_rng(0)
    cases = []
    # moderate magnitude case
    x1 = rng.uniform(-10.0, 10.0, size=20).astype(np.float64)
    c1 = rng.standard_normal()
    cases.append((x1, c1))
    # large magnitude case
    x2 = rng.uniform(-1000.0, 1000.0, size=30).astype(np.float64)
    c2 = rng.standard_normal()
    cases.append((x2, c2))

    max_err = 0.0
    for x, c in cases:
        try:
            x_list = x.tolist()
            x_shifted = (x + c).tolist()
            y1 = sol.logsumexp(x_list)
            y2 = sol.logsumexp(x_shifted) - c
        except Exception:
            return {"max_abs_err": float("inf")}
        ref1 = np.logaddexp.reduce(x)
        ref2 = np.logaddexp.reduce(x + c) - c
        err1 = abs(y1 - ref1)
        err2 = abs(y2 - ref2)
        max_err = max(max_err, err1, err2)

    return {"max_abs_err": max_err}
