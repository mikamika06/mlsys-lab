import numpy as np
from mlsys.scorers import rel_err

def grade(sol, fx) -> dict:
    # Test cases covering typical and edge situations.
    cases = [
        np.array([5, 3, 4]),
        np.array([1, 1, 1]),
        np.array([0, 0, 0]),
        np.arange(10),                     # lengths 0..9
        np.random.randint(1, 100, size=20) # random positive lengths
    ]
    max_err = 0.0
    for lengths in cases:
        try:
            got = sol.compute_padding_stats(lengths)
        except Exception:
            return {"rel_err": float("inf")}
        if not isinstance(got, (list, tuple)):
            return {"rel_err": float("inf")}
        got_arr = np.array(got, dtype=np.float64)
        batch = lengths.size
        max_len = int(np.max(lengths)) if batch > 0 else 0
        padded_ref = batch * max_len
        packed_ref = int(np.sum(lengths))
        waste_ref = (padded_ref - packed_ref) / padded_ref if padded_ref > 0 else 0.0
        ref_arr = np.array([padded_ref, packed_ref, waste_ref], dtype=np.float64)
        err = rel_err(ref_arr, got_arr)
        max_err = max(max_err, err)
    return {"rel_err": max_err}
