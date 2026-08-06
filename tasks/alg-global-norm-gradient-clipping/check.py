import numpy as np

def grade(sol, fx) -> dict:
    rng = np.random.default_rng(42)
    cases = [
        ([[3.0, 4.0]], 2.0),
        ([rng.normal(size=(10,)).tolist(), rng.normal(size=(5,)).tolist()], 1.0),
        ([ (rng.normal(size=(4,)) * 0.1).tolist(), (rng.normal(size=(3,)) * 0.1).tolist() ], 10.0),
        ([[0.0, 0.0], [0.0]], 1.0),
        ([rng.normal(size=(12,)).tolist(), rng.normal(size=(2,)).tolist()], 2.5),
    ]

    max_err = 0.0
    for grads, max_norm in cases:
        # ORACLE: Compute using exact formula via numpy
        np_grads = [np.array(g) for g in grads]
        total_norm_sq = 0.0
        for g in np_grads:
            total_norm_sq += np.sum(g**2)
        total_norm = np.sqrt(total_norm_sq)

        coef = min(1.0, max_norm / (total_norm + 1e-6))
        expected = [(g * coef).tolist() for g in np_grads]

        # Evaluate sol
        grads_copy = [list(g) for g in grads]
        try:
            ans = sol.clip_global_norm(grads_copy, max_norm)
            if not isinstance(ans, list) or len(ans) != len(expected):
                return {"max_abs_err": 1e9}

            for a, e in zip(ans, expected):
                if not isinstance(a, list) or len(a) != len(e):
                    return {"max_abs_err": 1e9}
                for va, ve in zip(a, e):
                    max_err = max(max_err, abs(va - ve))

        except Exception:
            return {"max_abs_err": float('inf')}

    return {"max_abs_err": float(max_err)}
