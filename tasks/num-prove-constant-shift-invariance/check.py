import numpy as np

def _stable_softmax(x):
    """Numerically stable softmax: subtract max, exponentiate, normalize."""
    x = np.asarray(x, dtype=np.float64)
    max_x = np.max(x, axis=-1, keepdims=True)
    e_x = np.exp(x - max_x)
    return e_x / np.sum(e_x, axis=-1, keepdims=True)

def grade(sol, fx) -> dict:
    """Grade by comparing user result against NumPy oracle reference."""
    test_cases = [
        ([1.0, 2.0, 3.0], [0.0, 0.0, 0.0]),
        ([1.0, 2.0, 3.0], [100.0, 100.0, 100.0]),
        ([10.0, 20.0, 30.0], [-50.0, -50.0, -50.0]),
        ([[1.0, 2.0], [3.0, 4.0]],
         [[10.0, 10.0], [10.0, 10.0]]),
        ([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]],
         [[100.0, 100.0, 100.0], [-100.0, -100.0, -100.0]]),
    ]

    max_abs_err = 0.0
    for logits, shift in test_cases:
        try:
            # --- reference answer via NumPy oracle ---
            logits_np = np.asarray(logits, dtype=np.float64)
            shift_np = np.asarray(shift, dtype=np.float64)
            ref_soft = _stable_softmax(logits_np)
            ref_shifted = _stable_softmax(logits_np - shift_np)
            ref_err = float(np.max(np.abs(ref_soft - ref_shifted)))

            # --- user answer ---
            user_err = sol.softmax_shift_invariant(logits, shift)
            user_err = float(user_err)

            # the metric is the difference between user and oracle
            err = abs(user_err - ref_err)
            if err > max_abs_err:
                max_abs_err = err
        except Exception:
            return {"max_abs_err": 1.0}

    return {"max_abs_err": max_abs_err}
