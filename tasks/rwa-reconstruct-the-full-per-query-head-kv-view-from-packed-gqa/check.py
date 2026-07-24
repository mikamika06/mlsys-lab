import numpy as np

def grade(sol, fx) -> dict:
    # Reference implementation uses np.repeat along axis 0
    def ref(kv_packed, n_rep):
        return np.repeat(kv_packed, n_rep, axis=0)

    cases = [
        (np.array([[[1], [2]]]), 3),                     # H=1, L=2, D=1
        (np.random.randint(0, 10, (4, 5, 6)), 2),        # random small tensor
        (np.arange(24).reshape((2, 3, 4)), 4),           # deterministic values
        (np.zeros((1, 1, 1)), 5),                       # all zeros
    ]

    ok = 1.0
    for kv_packed, n_rep in cases:
        try:
            got = sol.unpack_gqa(kv_packed, n_rep)
            expected = ref(kv_packed, n_rep)
        except Exception:
            return {"exact_match": 0.0}
        if not np.array_equal(got, expected):
            return {"exact_match": 0.0}
    return {"exact_match": ok}
