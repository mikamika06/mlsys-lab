import numpy as np


def _quant_block_rows(W, qmax):
    amax = np.max(np.abs(W), axis=1)
    d = np.where(amax > 0, amax / qmax, 1.0)
    codes = np.clip(np.round(W / d[:, None]), -qmax, qmax)
    return codes * d[:, None]


def q4_q8_reconstruction_mse(W: np.ndarray):
    """
    Blockwise ggml-style symmetric quantization, one block per row
    (block size = row length):

    - Q4_0: signed 4-bit codes, range [-8, 7], scale d = max(|row|) / 8.
    - Q8_0: signed 8-bit codes, range [-127, 127], scale d = max(|row|) / 127.

    Both use round-to-nearest with the per-row absmax scale (no zero-point
    -- symmetric quantization). Returns (mse_q4_0, mse_q8_0): the mean
    squared reconstruction error over every element of W, for each format.
    """
    W = np.asarray(W, dtype=np.float64)
    q4 = _quant_block_rows(W, 8)
    q8 = _quant_block_rows(W, 127)
    mse4 = float(np.mean((q4 - W) ** 2))
    mse8 = float(np.mean((q8 - W) ** 2))
    return mse4, mse8
