def _reference(A):
    import numpy as np
    U, S, Vt = np.linalg.svd(A, full_matrices=False)
    energy = S**2
    cum_energy = np.cumsum(energy)
    total = cum_energy[-1]
    ratio = cum_energy / total
    k = int(np.searchsorted(ratio, 0.9) + 1)
    size_ratio = (A.shape[0]*k + k + k*A.shape[1])/(A.shape[0]*A.shape[1])
    return k, float(size_ratio)

def grade(sol, fx):
    import numpy as np
    cases = [
        np.random.randn(10, 5),
        np.random.randn(50, 20),
        np.random.randn(30, 30),
        np.random.randn(100, 15),
        np.random.randn(25, 40)
    ]
    ok = 1.0
    max_rel_err = 0.0
    for A in cases:
        try:
            k_out, size_ratio_out = sol.pick_rank_and_report(A)
        except Exception:
            return {"exact_match": 0.0, "rel_err": float("inf")}
        ref_k, ref_size_ratio = _reference(A)
        if k_out != ref_k:
            ok = 0.0
        rel_err = abs(size_ratio_out - ref_size_ratio) / (abs(ref_size_ratio)+1e-12)
        if rel_err > max_rel_err:
            max_rel_err = rel_err
    return {"exact_match": ok, "rel_err": max_rel_err}
