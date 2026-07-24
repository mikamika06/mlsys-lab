import numpy as np


def grade(sol, fx) -> dict:
    cases = [
        (
            np.array([0.0, 1.0, 1000.0, -1000.0, 50.0]),
            np.array([0.0, 2.0, 999.0, -999.0, 49.0]),
        ),
        (
            np.array([[3.0, -4.0], [700.0, 701.0]]),
            np.array([[2.0, -3.5], [699.0, 710.0]]),
        ),
        (
            np.array([-1000.0, -1001.0, 1e-8]),
            np.array([-999.0, -1002.0, 2e-8]),
        ),
    ]

    refs = []
    outs = []
    try:
        for a, b in cases:
            refs.append(np.logaddexp(a, b))
            outs.append(np.asarray(sol.stable_log_add_exp(a, b), dtype=np.float64))
    except Exception:
        return {"rel_err": float("inf")}

    ref = np.concatenate([x.ravel() for x in refs])
    out = np.concatenate([x.ravel() for x in outs])

    if ref.shape != out.shape or not np.all(np.isfinite(out)):
        return {"rel_err": float("inf")}

    err = float(np.linalg.norm(out - ref) / (np.linalg.norm(ref) + 1e-12))
    return {"rel_err": err}
