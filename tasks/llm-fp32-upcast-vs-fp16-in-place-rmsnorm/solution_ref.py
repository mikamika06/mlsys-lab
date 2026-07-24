import numpy as np

def rmsnorm(x: np.ndarray, *, upcast: bool = True) -> np.ndarray:
    """
    Compute RMSNorm of a float16 array with optional float32 upcast for the reduction.
    
    Parameters
    ----------
    x : np.ndarray
        Input array of dtype float16 and shape (n, d).
    upcast : bool, default=True
        If True, cast to float32 before computing the mean of squares.
        
    Returns
    -------
    y : np.ndarray
        RMS-normalized array with dtype float16.
    """
    if upcast:
        # Cast to float32 for reduction
        y = x.astype(np.float32)
        mean_sq = np.mean(y * y, axis=-1, keepdims=True)
        rms = np.sqrt(mean_sq).astype(np.float32)
        out = (x / rms).astype(np.float16)
    else:
        # Pure float16 computation
        mean_sq = np.mean(x * x, axis=-1, keepdims=True)
        rms = np.sqrt(mean_sq)
        out = (x / rms).astype(np.float16)
    return out
