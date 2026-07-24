import numpy as np

def per_channel_scales(X: np.ndarray) -> np.ndarray:
    """TODO: This implementation incorrectly uses the mean absolute value instead of RMS.
It will fail the channel_rel_err gate because the scale is too small."""
    raise NotImplementedError('your code here')
