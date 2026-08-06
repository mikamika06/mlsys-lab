import numpy as np

def int2_quant_with_residual(x: np.ndarray, R: int):
    """
    Quantize all but the last R tokens to 2‑bit integer codes and keep the
    most recent R tokens in fp16 precision.
    
    Parameters
    ----------
    x : array_like
        One‑dimensional sequence of real numbers.
    R : int
        Number of most recent tokens to preserve in full precision (fp16).
    
    Returns
    -------
    codes : np.ndarray, dtype=uint8
        2‑bit integer codes for the first len(x)-R elements. Empty if R >= len(x).
    residuals : np.ndarray, dtype=float16
        The last R elements of x cast to float16.
    """
    x = np.asarray(x, dtype=np.float32)
    n = len(x)
    if R >= n:
        return np.empty((0,), dtype=np.uint8), x.astype(np.float16)
    seg = x[:n-R]
    min_val = seg[0]
    max_val = seg[0]
    for i in range(len(seg)):
        val = seg[i]
        if val < min_val:
            min_val = val
        if val > max_val:
            max_val = val
    if min_val == max_val:
        codes = np.zeros(len(seg), dtype=np.uint8)
    else:
        scale = 3.0 / (max_val - min_val)
        codes_list = []
        for i in range(len(seg)):
            val = seg[i]
            r = round((val - min_val) * scale)
            if r < 0:
                r = 0
            elif r > 3:
                r = 3
            codes_list.append(r)
        codes = np.array(codes_list, dtype=np.uint8)
    residuals = x[n-R:].astype(np.float16)
    return codes, residuals
