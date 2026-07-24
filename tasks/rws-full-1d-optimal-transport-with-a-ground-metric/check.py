import numpy as np
from scipy.optimize import linprog


def _lp_ot_cost(positions: np.ndarray, p: np.ndarray, q: np.ndarray) -> float:
    """Exact transportation-LP optimal cost, independent of the
    cumulative-sum closed form -- the real oracle for this task."""
    n = positions.shape[0]
    C = np.abs(positions[:, None] - positions[None, :])
    c = C.reshape(-1)

    A_eq = []
    b_eq = []
    for i in range(n):
        row = np.zeros((n, n))
        row[i, :] = 1.0
        A_eq.append(row.reshape(-1))
        b_eq.append(p[i])
    for j in range(n):
        col = np.zeros((n, n))
        col[:, j] = 1.0
        A_eq.append(col.reshape(-1))
        b_eq.append(q[j])

    res = linprog(c, A_eq=np.array(A_eq), b_eq=np.array(b_eq), bounds=(0, None), method="highs")
    if not res.success:
        raise RuntimeError("LP oracle failed to solve")
    return float(res.fun)


def _build_cases():
    cases = []
    for seed, n in [(0, 5), (1, 6), (2, 8), (3, 4)]:
        rng = np.random.default_rng(seed)
        positions = rng.uniform(-10.0, 10.0, size=n)
        while len(set(np.round(positions, 9))) != n:  # ensure distinct
            positions = rng.uniform(-10.0, 10.0, size=n)
        p = rng.uniform(0.1, 1.0, size=n)
        q = rng.uniform(0.1, 1.0, size=n)
        p = p / p.sum()
        q = q / q.sum()
        cases.append((positions, p, q))
    return cases


def grade(sol, fx) -> dict:
    worst_rel = 0.0
    for positions, p, q in _build_cases():
        ref = _lp_ot_cost(positions, p, q)

        try:
            got = float(sol.ot_cost_1d(positions.copy(), p.copy(), q.copy()))
        except Exception:
            return {"rel_err": float("inf")}

        if not np.isfinite(got):
            return {"rel_err": float("inf")}

        rel = abs(got - ref) / (abs(ref) + 1e-12)
        worst_rel = max(worst_rel, rel)

    return {"rel_err": worst_rel}
