import numpy as np

def _ref_count(x: np.ndarray) -> int:
    """Oracle that counts non‑finite entries produced by a naïve softmax."""
    exp_x = np.exp(x, dtype=np.float64)
    sum_exp = np.sum(exp_x, axis=1, keepdims=True)
    softmax = exp_x / sum_exp
    return int(np.count_nonzero(~np.isfinite(softmax)))

def grade(sol, fx) -> dict:
    # Test case that triggers overflow in the naïve softmax.
    X = np.array([[1000, 1000, 1000],
                  [2000, 2000, 2000]], dtype=np.float64)

    expected = _ref_count(X)
    try:
        got = sol.count_nonfinite_in_naive_softmax(X)
    except Exception:
        return {"exact_match": 0.0}

    ok = 1.0 if int(got) == expected else 0.0
    return {"exact_match": ok}
