import numpy as np


def _oracle(ns):
    out = []
    for n in ns:
        idx = np.arange(n, dtype=np.float64)
        H = 1.0 / (idx[:, None] + idx[None, :] + 1.0)
        singular_values = np.linalg.svd(H, compute_uv=False)
        out.append(np.log10(singular_values[0] / singular_values[-1]))
    return np.asarray(out, dtype=np.float64)


def grade(sol, fx) -> dict:
    ns = [2, 3, 5, 8, 10, 12]
    ref = _oracle(ns)
    try:
        got = np.asarray(sol.hilbert_condition_numbers(ns), dtype=np.float64)
    except Exception:
        return {"rel_err": float("inf")}

    if got.shape != ref.shape or not np.all(np.isfinite(got)):
        return {"rel_err": float("inf")}

    err = float(np.mean(np.abs(got - ref) / (np.abs(ref) + 1e-12)))
    return {"rel_err": err}
