import numpy as np

def compare_rounding(x: np.ndarray):
    """Incorrect implementation that uses the same FP16 rounding for both outputs.
This will cause the BF16 metric to fail."""
    raise NotImplementedError('your code here')
