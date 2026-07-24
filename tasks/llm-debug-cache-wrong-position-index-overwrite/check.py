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
            np.arange(20, dtype=np.float64).reshape(5, 4),
            np.arange(100, 120, dtype=np.float64).reshape(5, 4),
            np.array([9.0, 8.0, 7.0, 6.0]),
            np.array([5.0, 4.0, 3.0, 2.0]),
            0,
        ),
        (
            np.zeros((6, 3), dtype=np.float64),
            np.ones((6, 3), dtype=np.float64),
            np.array([1.5, -2.0, 3.5]),
            np.array([-1.0, 2.0, -3.0]),
            3,
        ),
        (
            np.full((8, 2), 7.0, dtype=np.float64),
            np.full((8, 2), -4.0, dtype=np.float64),
            np.array([11.0, 12.0]),
            np.array([13.0, 14.0]),
            7,
        ),
    ]

    worst = 0.0
    for cache_k, cache_v, new_k, new_v, position in cases:
        ref_k, ref_v = _oracle(cache_k, cache_v, new_k, new_v, position)
        try:
            got_k, got_v = sol.write_kv_cache(
                cache_k.copy(),
                cache_v.copy(),
                new_k.copy(),
                new_v.copy(),
                position,
            )
            err = max(_max_abs_err(got_k, ref_k), _max_abs_err(got_v, ref_v))
        except Exception:
            return {"max_abs_err": float("inf")}
        worst = max(worst, err)
    return {"max_abs_err": worst}
