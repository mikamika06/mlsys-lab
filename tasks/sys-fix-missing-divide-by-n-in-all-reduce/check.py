import numpy as np


def grade(sol, fx) -> dict:
    rng = np.random.default_rng(0)
    worst = 0.0
    for _ in range(6):
        N = int(rng.integers(2, 9))
        shape = tuple(int(x) for x in rng.integers(1, 6, size=rng.integers(1, 3)))
        grads = [rng.standard_normal(shape) for _ in range(N)]
        expected = np.mean(np.stack(grads, axis=0), axis=0)
        try:
            got = np.asarray(sol.all_reduce_mean_grads([g.copy() for g in grads]), dtype=np.float64)
            if got.shape != expected.shape:
                worst = float("inf")
                break
            err = float(np.linalg.norm(got - expected) / (np.linalg.norm(expected) + 1e-12))
        except Exception:
            worst = float("inf")
            break
        worst = max(worst, err)
    return {"rel_err": worst}
