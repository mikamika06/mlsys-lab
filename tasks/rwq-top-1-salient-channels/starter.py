import math

def top_salient_channels(X: list[list[float]], frac: float=0.01) -> list[int]:
    """AWQ-style salient-channel selection: the top `frac` fraction of
    channels (columns) by mean absolute calibration activation.

    Returns a 1-D int array of channel indices, length ceil(frac * C)
    (minimum 1), the channels with the largest mean(|X|).
    """
    raise NotImplementedError('your code here')
