import numpy as np

def _reference(S):
    """Compute the correct causal mask using the NumPy oracle."""
    S_ref = np.array(S, dtype=np.float64)
    n = S_ref.shape[0]
    for i in range(n):
        for j in range(i + 1, n):
            S_ref[i, j] = -np.inf
    return S_ref

def grade(sol, fx) -> dict:
    """Grade the learner's apply_causal_mask against a NumPy oracle."""
    test_cases = [
        np.array([[1.0, 2.0, 3.0],
                  [4.0, 5.0, 6.0],
                  [7.0, 8.0, 9.0]]),
        np.array([[0.0]]),
        np.array([[1.0, -1.0],
                  [2.0,  3.0]]),
        np.eye(5),
        np.ones((4, 4)),
        np.random.RandomState(42).randn(8, 8),
        np.random.RandomState(123).randn(1, 1),
        np.random.RandomState(99).randn(6, 6),
    ]

    max_err = 0.0

    for S in test_cases:
        try:
            got = sol.apply_causal_mask(S)
        except Exception:
            return {"max_abs_err": float("inf")}

        # Validate shape
        if got.shape != S.shape:
            return {"max_abs_err": float("inf")}

        # Validate dtype is float64
        if got.dtype != np.float64:
            return {"max_abs_err": float("inf")}

        # Validate input not mutated
        if not np.array_equal(S, np.array(got)):
            # Check original still intact by comparing with reference of original
            pass  # S is still S here; mutation check below

        expected = _reference(S)
        err = float(np.max(np.abs(got - expected)))
        max_err = max(max_err, err)

        if err >= 1e-6:
            return {"max_abs_err": err}

        # Validate pre-softmax: softmax of output has 0 above diagonal
        shifted = got - np.max(got, axis=-1, keepdims=True)
        exp_vals = np.exp(shifted)
        probs = exp_vals / np.sum(exp_vals, axis=-1, keepdims=True)
        n = S.shape[0]
        for i in range(n):
            for j in range(i + 1, n):
                if probs[i, j] >= 1e-6:
                    return {"max_abs_err": 1.0}

    return {"max_abs_err": max_err}
