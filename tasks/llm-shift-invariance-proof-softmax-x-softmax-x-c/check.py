import numpy as np

def _np_softmax(x):
    """NumPy reference: numerically stable softmax via max-subtraction."""
    x = np.asarray(x, dtype=np.float64)
    e = np.exp(x - np.max(x))
    return e / np.sum(e)

def grade(sol, fx) -> dict:
    test_vectors = [
        np.array([1.0, 2.0, 3.0]),
        np.array([0.0, 0.0, 0.0]),
        np.array([1000.0, 1001.0, 1002.0]),
        np.array([-500.0, -499.0, -498.0]),
        np.array([1e6, 1e6, 1e6]),
        np.array([-10.0, -5.0, 0.0, 5.0, 10.0]),
        np.array([500.0, 500.0, 500.0, 500.0]),
    ]

    shifts = [0.0, 1.0, -1.0, 100.0, -100.0, 1000.0, -1000.0]

    accuracy_max_err = 0.0
    shift_max_err = 0.0

    for x in test_vectors:
        # --- accuracy gate ---
        try:
            got = np.asarray(sol.softmax(x), dtype=np.float64)
            ref = _np_softmax(x)
            err = float(np.max(np.abs(got - ref)))
        except Exception:
            return {"accuracy": float("inf"), "shift_invariance": float("inf")}
        accuracy_max_err = max(accuracy_max_err, err)

        # --- shift-invariance gate ---
        try:
            s1 = np.asarray(sol.softmax(x), dtype=np.float64)
        except Exception:
            return {"accuracy": float("inf"), "shift_invariance": float("inf")}
        for c in shifts:
            try:
                s2 = np.asarray(sol.softmax(x - c), dtype=np.float64)
                err = float(np.max(np.abs(s1 - s2)))
            except Exception:
                return {"accuracy": float("inf"), "shift_invariance": float("inf")}
            shift_max_err = max(shift_max_err, err)

    return {"accuracy": accuracy_max_err, "shift_invariance": shift_max_err}
