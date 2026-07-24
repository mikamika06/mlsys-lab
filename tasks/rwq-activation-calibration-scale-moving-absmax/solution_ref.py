import numpy as np

def moving_absmax(batches, momentum):
    """
    Correct implementation of the moving absolute maximum calibration.
    """
    scale = np.zeros(batches[0].shape[1], dtype=np.float64)
    for batch in batches:
        absmax = np.max(np.abs(batch), axis=0)
        scale = momentum * scale + (1 - momentum) * absmax
    return scale
