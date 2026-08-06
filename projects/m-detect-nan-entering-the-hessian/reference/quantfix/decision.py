import numpy as np
from quantfix.hessian import validate_hessian


def should_use_model_free_ptq(h, threshold=1e6):
    """Decide whether model-free PTQ is required based on Hessian validity and conditioning."""
    if not validate_hessian(h):
        return True
    try:
        cond = np.linalg.cond(h)
        if np.isnan(cond) or cond > threshold:
            return True
    except Exception:
        return True
    return False
