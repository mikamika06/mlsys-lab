import numpy as np
from mlsys import scorers

def _ref_stable_softmax(logits: np.ndarray) -> np.ndarray:
    """Reference: numerically stable one-shot softmax via NumPy."""
    logits = np.asarray(logits, dtype=np.float64)
    m = np.max(logits)
    ex = np.exp(logits - m)
    return ex / np.sum(ex)

def grade(sol, fx) -> dict:
    """Grade the student's blockwise_softmax against a NumPy oracle."""
    test_cases = [
        (np.array([2.0, -1.0, 0.0, 3.0]), 2),
        (np.array([2.0, -1.0, 0.0, 3.0]), 1),
        (np.array([2.0, -1.0, 0.0, 3.0]), 4),
        (np.array([100.0, 200.0, 300.0]), 2),
        (np.array([-100.0, -200.0, -300.0]), 2),
        (np.linspace(-5, 5, 100), 10),
        (np.linspace(-10, 10, 200), 32),
        (np.linspace(-20, 20, 1000), 64),
    ]

    # Add fixture cases if available
    for fname, data in fx.items():
        try:
            import io
            with np.load(io.BytesIO(data)) as npz:
                test_cases.append((npz['logits'], int(npz['block_size'])))
        except Exception:
            pass

    max_err = 0.0
    for logits, block_size in test_cases:
        try:
            student = sol.blockwise_softmax(logits, block_size)
        except Exception:
            return {"max_abs_err": 1.0}

        student = np.asarray(student, dtype=np.float64)
        ref = _ref_stable_softmax(logits)

        if student.shape != logits.shape:
            return {"max_abs_err": 1.0}

        err = scorers.max_abs_err(ref, student)
        max_err = max(max_err, err)

    return {"max_abs_err": max_err}
