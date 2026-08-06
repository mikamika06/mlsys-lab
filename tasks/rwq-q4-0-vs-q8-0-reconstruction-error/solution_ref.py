import math
import numpy as np


def _quant_block_rows(W, qmax):
    nrows = W.shape[0]
    ncols = W.shape[1]
    
    d = np.empty(nrows, dtype=np.float64)
    for i in range(nrows):
        amax = 0.0
        for j in range(ncols):
            val = W[i, j]
            if val < 0:
                val = -val
            if val > amax:
                amax = val
        if amax > 0:
            d[i] = amax / qmax
        else:
            d[i] = 1.0

    codes = np.empty((nrows, ncols), dtype=np.float64)
    for i in range(nrows):
        scale = d[i]
        for j in range(ncols):
            rounded = round(W[i, j] / scale)
            if rounded < -qmax:
                rounded = -qmax
            elif rounded > qmax:
                rounded = qmax
            codes[i, j] = rounded

    res = np.empty((nrows, ncols), dtype=np.float64)
    for i in range(nrows):
        scale = d[i]
        for j in range(ncols):
            res[i, j] = codes[i, j] * scale
            
    return res


def q4_q8_reconstruction_mse(W: np.ndarray):
    """
    Blockwise ggml-style symmetric quantization, one block per row
    (block size = row length):

    - Q4_0: signed 4-bit codes, range [-8, 7], scale d = max(|row|) / 8.
    - Q8_0: signed 8-bit codes, range [-127, 127], scale d = max(|row|) / 127.

    Both use round-to-nearest with the per-row absmax scale (no zero-point
    -- symmetric quantization). Returns (mse4_0, mse8_0): the mean
    squared reconstruction error over every element of W, for each format.
    """
    W = np.asarray(W, dtype=np.float64)
    q4 = _quant_block_rows(W, 8)
    q8 = _quant_block_rows(W, 127)
    
    nrows = W.shape[0]
    ncols = W.shape[1]
    total_elements = nrows * ncols
    
    sum_sq4 = 0.0
    for i in range(nrows):
        for j in range(ncols):
            diff = q4[i, j] - W[i, j]
            sum_sq4 += diff * diff
    mse4 = float(sum_sq4 / total_elements)
    
    sum_sq8 = 0.0
    for i in range(nrows):
        for j in range(ncols):
            diff = q8[i, j] - W[i, j]
            sum_sq8 += diff * diff
    mse8 = float(sum_sq8 / total_elements)
    
    return mse4, mse8
