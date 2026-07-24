import itertools
import numpy as np


def _oracle(W):
    patterns = list(itertools.combinations(range(4), 2))
    out = []
    for row in np.asarray(W, dtype=np.float64):
        best = None
        for keep in patterns:
            dropped = 0.0
            for i in range(4):
                if i not in keep:
                    dropped += abs(row[i])
            if best is None or dropped < best:
                best = dropped
        out.append(best)
    return np.asarray(out, dtype=np.float64)


def grade(sol, fx) -> dict:
    rng = np.random.default_rng(12345)
    W = rng.normal(size=(32, 4))
    W[0] = np.array([5.0, -1.0, 2.0, -7.0])
    W[1] = np.array([0.0, -0.0, 3.0, -3.0])
    ref = _oracle(W)
    try:
        got = np.asarray(sol.greedy_24_prune(W), dtype=np.float64)
    except Exception:
        return {"exact_match": 0.0}
    ok = (
        got.shape == ref.shape
        and np.allclose(got, ref, rtol=0.0, atol=1e-9)
    )
    return {"exact_match": 1.0 if ok else 0.0}
