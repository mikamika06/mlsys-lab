import numpy as np

def _reference(local_arrays):
    """Compute the correct all‑gather buffer."""
    return np.concatenate([np.asarray(a, dtype=np.float64) for a in local_arrays])

def grade(sol, fx) -> dict:
    # Test cases with varying sizes and values
    rng = np.random.default_rng(0)
    ok = 1.0
    max_err = 0.0

    for n_ranks in [1, 2, 3, 5]:
        m_local = rng.integers(1, 10)  # local size per rank
        local_arrays = [rng.random(m_local).astype(np.float64) for _ in range(n_ranks)]
        ref = _reference(local_arrays)

        try:
            out = sol.ring_all_gather(list(local_arrays))
        except Exception:
            return {"max_abs_err": float("inf")}

        if not isinstance(out, (list, tuple)) or len(out) != n_ranks:
            return {"max_abs_err": float("inf")}

        for rank_arr in out:
            if not isinstance(rank_arr, np.ndarray):
                return {"max_abs_err": float("inf")}
            err = np.max(np.abs(rank_arr.astype(np.float64) - ref))
            max_err = max(max_err, err)

    return {"max_abs_err": max_err}
