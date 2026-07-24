import numpy as np
from mlsys import scorers


def _oracle(L, N, C):
    eager = N * L + C
    graph = L + C
    frac = (eager - graph) / eager
    return np.array([eager, graph, frac], dtype=np.float64)


def _cases():
    cases = [
        (0.02, 1, 5.0),      # single kernel: graph gives zero savings
        (0.02, 200, 5.0),
        (0.1, 50, 0.5),      # launch-overhead-dominated step
        (1e-4, 4000, 12.0),  # many tiny kernels
        (0.5, 10, 1000.0),   # compute-dominated step
    ]
    rng = np.random.default_rng(0)
    for _ in range(8):
        L = float(rng.uniform(1e-5, 2.0))
        N = int(rng.integers(1, 5000))
        C = float(rng.uniform(0.0, 2000.0))
        cases.append((L, N, C))
    return cases


def grade(sol, fx) -> dict:
    worst = 0.0
    for L, N, C in _cases():
        ref = _oracle(L, N, C)
        try:
            got = np.asarray(sol.graph_launch_step_time(L, N, C), dtype=np.float64)
        except Exception:
            return {"rel_err": float("inf")}
        if got.shape != ref.shape:
            return {"rel_err": float("inf")}
        worst = max(worst, scorers.rel_err(ref, got))
    return {"rel_err": worst}
