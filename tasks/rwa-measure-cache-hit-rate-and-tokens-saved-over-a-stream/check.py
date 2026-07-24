import numpy as np
from mlsys import scorers


def _oracle(requests):
    seen = []
    hits = 0
    reused = 0
    for req in requests:
        hit = False
        for old in seen:
            if old == req:
                hit = True
                break
        if hit:
            hits += 1
            reused += len(req)
        seen.append(list(req))
    rate = hits / len(requests) if requests else 0.0
    return rate, reused


def grade(sol, fx) -> dict:
    cases = [
        [
            [1, 2, 3],
            [4, 5],
            [1, 2, 3],
            [4, 5],
            [6],
        ],
        [
            [7, 7],
            [7],
            [7, 7],
            [7],
            [7, 7],
        ],
        [
            [10, 11],
            [11, 10],
            [10, 11],
            [12, 13, 14],
            [12, 13, 14],
            [11, 10],
        ],
        [],
    ]

    max_err = 0.0
    for stream in cases:
        try:
            got = sol.measure_cache_stats([list(x) for x in stream])
        except Exception:
            return {"rel_err": 1.0}
        ref = _oracle(stream)
        err = scorers.rel_err(
            np.asarray(ref, dtype=np.float64),
            np.asarray(got, dtype=np.float64),
        )
        max_err = max(max_err, err)
    return {"rel_err": max_err}
