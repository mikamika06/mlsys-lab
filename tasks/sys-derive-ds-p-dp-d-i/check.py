import numpy as np


def _oracle(P, dP):
    rows = []
    for p, g in zip(P, dP):
        J = np.diag(p) - np.outer(p, p)
        rows.append(J @ g)
    return np.asarray(rows, dtype=np.float64)


def _rel_err(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-12))


def grade(sol, fx) -> dict:
    rng = np.random.default_rng(7)
    cases = []
    for n, m in [(2, 3), (4, 5), (8, 8)]:
        logits = rng.normal(size=(n, m))
        logits -= np.max(logits, axis=1, keepdims=True)
        exp = np.exp(logits)
        P = exp / np.sum(exp, axis=1, keepdims=True)
        dP = rng.normal(size=(n, m))
        cases.append((P.astype(np.float64), dP.astype(np.float64)))

    score = 0.0
    for P, dP in cases:
        try:
            got = sol.derive_ds(P, dP)
        except Exception:
            return {"rel_err": float("inf")}
        ref = _oracle(P, dP)
        err = _rel_err(got, ref)
        score = max(score, err)
    return {"rel_err": score}
