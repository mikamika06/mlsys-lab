import numpy as np


def _oracle(cache_k, cache_v, new_k, new_v, position):
    ref_k = np.array(cache_k, dtype=np.float64, copy=True)
    ref_v = np.array(cache_v, dtype=np.float64, copy=True)
    ref_k[position, :] = new_k
    ref_v[position, :] = new_v
    return ref_k, ref_v


def _max_abs_err(a, b):
    return float(np.max(np.abs(np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64))))


def grade(sol, fx) -> dict:
    cases = [
        (
            [[float(i) for i in range(j, j + 4)] for j in range(0, 20, 4)],
            [[float(i) for i in range(j, j + 4)] for j in range(100, 120, 4)],
            [9.0, 8.0, 7.0, 6.0],
            [5.0, 4.0, 3.0, 2.0],
            0,
        ),
        (
            [[0.0]*3 for _ in range(6)],
            [[1.0]*3 for _ in range(6)],
            [1.5, -2.0, 3.5],
            [-1.0, 2.0, -3.0],
            3,
        ),
        (
            [[7.0]*2 for _ in range(8)],
            [[-4.0]*2 for _ in range(8)],
            [11.0, 12.0],
            [13.0, 14.0],
            7,
        ),
    ]

    worst = 0.0
    for cache_k, cache_v, new_k, new_v, position in cases:
        ref_k, ref_v = _oracle(cache_k, cache_v, new_k, new_v, position)
        try:
            got_k, got_v = sol.write_kv_cache(
                [row[:] for row in cache_k],
                [row[:] for row in cache_v],
                list(new_k),
                list(new_v),
                position,
            )
            err = max(_max_abs_err(got_k, ref_k), _max_abs_err(got_v, ref_v))
        except Exception:
            return {"max_abs_err": float("inf")}
        worst = max(worst, err)
    return {"max_abs_err": worst}
