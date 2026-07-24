import numpy as np


def calibrate_and_score(X: np.ndarray, num_bits: int = 8) -> dict:
    """Score three int8 calibration methods on X by symmetric-quantization MSE.

    Returns {'minmax': mse, 'percentile': mse, 'entropy': mse}, each the mean
    squared reconstruction error of round-tripping X through symmetric linear
    quantization with that method's clipping threshold. See task.md for the
    exact per-method threshold rule.
    """
    raise NotImplementedError('your code here')
