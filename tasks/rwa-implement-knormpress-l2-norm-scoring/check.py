import numpy as np

def _reference_knormpress(data, budget):
    norms = {k: np.linalg.norm(v) for k, v in data.items()}
    sorted_keys = sorted(norms, key=norms.get, reverse=True)
    top_k = sorted(sorted_keys[:budget])
    return top_k

def grade(sol, fx) -> dict:
    cases = [
        ({1: np.array([3., 4.]), 2: np.array([0., 0.])}, 1),
        ({10: np.array([3., 4.]), 20: np.array([6., 8.]), 30: np.array([1., 1.])}, 2),
        ({5: np.array([0., 0.]), 7: np.array([0., 0.])}, 5),
        ({}, 3),
        ({42: np.array([2., -2.])}, 0)
    ]
    ok = 1.0
    for data, budget in cases:
        try:
            got = sol.knormpress(data, budget)
            if not isinstance(got, list):
                ok = 0.0
                break
            expected = _reference_knormpress(data, budget)
            if got != expected:
                ok = 0.0
                break
        except Exception:
            ok = 0.0
            break
    return {"exact_match": ok}
