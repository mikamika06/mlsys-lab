import numpy as np

def _reference(W):
    zeros = int(np.count_nonzero(W == 0))
    ratio = zeros / W.size if W.size else 0.0
    return zeros, float(ratio)

def grade(sol, fx) -> dict:
    cases = [
        (np.array([1., 2., 3.]), "dense"),
        (np.array([0., 0., 5., 6.]), "partial zeros"),
        (np.zeros((4,)), "all zeros"),
        (np.arange(12).reshape(3,4), "no zeros"),
    ]
    ok = 1.0
    for W, name in cases:
        try:
            got = sol.count_zeros_and_sparsity(W)
        except Exception:
            return {"exact_match": 0.0}
        ref = _reference(W)
        if not isinstance(got, tuple) or len(got) != 2:
            ok = 0.0
            break
        # compare integers exactly
        if got[0] != ref[0]:
            ok = 0.0
            break
        # compare floats with tolerance
        if abs(got[1] - ref[1]) > 1e-12:
            ok = 0.0
            break
    return {"exact_match": ok}
