import ref
import numpy as np


def check(workdir):
    from gradclip.loop import run_training_step
    out = {"nan_prevented": 0.0}
    weights = np.array([1.0, 2.0], dtype=np.float32)
    grad = np.array([1e10, 1e10], dtype=np.float32)
    try:
        _, got_nan = run_training_step(weights, grad, 1.0, 0.1)
        _, want_nan = ref.run_training_step(weights, grad, 1.0, 0.1)
        if got_nan == want_nan and not got_nan:
            out["nan_prevented"] = 1.0
        else:
            out["_note"] = f"expected nan prevented to be True, got nan={got_nan}"
    except Exception as e:
        out["_note"] = f"error during execution: {str(e)}"
    return out
