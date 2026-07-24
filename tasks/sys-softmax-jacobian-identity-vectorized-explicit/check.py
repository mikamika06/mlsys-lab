import tracemalloc
import numpy as np


def _oracle(p, dY):
    rows = []
    for pi, gi in zip(p, dY):
        J = np.diag(pi) - np.outer(pi, pi)
        rows.append(J @ gi)
    return np.asarray(rows, dtype=np.float64)


def grade(sol, fx) -> dict:
    cases = [
        (
            np.array(
                [
                    [0.2, 0.3, 0.5],
                    [0.7, 0.2, 0.1],
                ],
                dtype=np.float64,
            ),
            np.array(
                [
                    [1.0, -2.0, 0.5],
                    [0.1, 0.8, -0.4],
                ],
                dtype=np.float64,
            ),
        ),
        (
            np.array(
                [
                    [0.05, 0.15, 0.25, 0.55],
                    [0.4, 0.3, 0.2, 0.1],
                    [0.1, 0.1, 0.2, 0.6],
                ],
                dtype=np.float64,
            ),
            np.array(
                [
                    [2.0, 1.0, -1.0, 3.0],
                    [-2.0, 0.5, 4.0, 1.0],
                    [0.0, 1.0, 1.5, -0.5],
                ],
                dtype=np.float64,
            ),
        ),
    ]

    err = 0.0
    for p, dY in cases:
        try:
            got = np.asarray(sol.softmax_jacobian_vjp(p, dY), dtype=np.float64)
        except Exception:
            return {"rel_err": float("inf"), "no_full_jacobian": 0.0}
        ref = _oracle(p, dY)
        err = max(err, float(np.linalg.norm(got - ref) / (np.linalg.norm(ref) + 1e-12)))

    p_big = np.full((4, 1024), 1.0 / 1024.0, dtype=np.float64)
    d_big = np.arange(4096, dtype=np.float64).reshape(4, 1024)
    tracemalloc.start()
    try:
        sol.softmax_jacobian_vjp(p_big, d_big)
        _, peak = tracemalloc.get_traced_memory()
    except Exception:
        peak = 10**9
    finally:
        tracemalloc.stop()

    return {
        "rel_err": err,
        "no_full_jacobian": 1.0 if peak < 20_000_000 else 0.0,
    }
