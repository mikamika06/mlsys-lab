import numpy as np


def top_salient_channels(X: np.ndarray, frac: float = 0.01) -> np.ndarray:
    """AWQ-style salient-channel selection: the top `frac` fraction of
    channels (columns) by mean absolute calibration activation.

    Returns a 1-D int array of channel indices, length ceil(frac * C)
    (minimum 1), the channels with the largest mean(|X|).
    """
    raise NotImplementedError('your code here')
