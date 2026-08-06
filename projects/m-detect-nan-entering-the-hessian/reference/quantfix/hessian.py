import numpy as np


def validate_hessian(h):
    """Validate that the Hessian matrix contains no NaNs or Infs."""
    arr = np.asarray(h, dtype=np.float32)
    if not np.isfinite(arr).all():
        return False
    return True
