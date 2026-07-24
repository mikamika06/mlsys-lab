import numpy as np

def classify_rounding(W: np.ndarray, V: np.ndarray, s: float) -> np.ndarray:
    """Broken implementation: treats any difference as an upward rounding.
This will fail the exact_match gate because it never reports -1 or 0."""
    raise NotImplementedError('your code here')
