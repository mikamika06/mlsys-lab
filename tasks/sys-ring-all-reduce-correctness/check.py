import numpy as np


def _gen_case(rng):
    n_ranks = int(rng.integers(2, 6))
    chunks_per_rank = int(rng.integers(1, 4))
    L = n_ranks * chunks_per_rank
    buffers = [rng.standard_normal(L) for _ in range(n_ranks)]
    return buffers


def grade(sol, fx) -> dict:
    rng = np.random.default_rng(0)
    cases = [_gen_case(rng) for _ in range(8)]
    # explicit small fixed case (N=4 ranks, L=4 -> chunk size 1)
    cases.append([np.array([1.0, 2.0, 3.0, 4.0]),
                  np.array([10.0, 20.0, 30.0, 40.0]),
                  np.array([100.0, 200.0, 300.0, 400.0]),
                  np.array([1000.0, 2000.0, 3000.0, 4000.0])])

    worst = 0.0
    for buffers in cases:
        expected = np.sum(np.stack(buffers, axis=0), axis=0)
        try:
            got = sol.ring_all_reduce([b.copy() for b in buffers])
            if len(got) != len(buffers):
                worst = float("inf")
                break
            for g in got:
                g = np.asarray(g, dtype=np.float64)
                if g.shape != expected.shape:
                    worst = float("inf")
                    break
                worst = max(worst, float(np.max(np.abs(g - expected))))
        except Exception:
            worst = float("inf")
            break
    return {"max_abs_err": worst}
