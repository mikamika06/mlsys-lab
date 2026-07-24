import numpy as np


def q4_q8_reconstruction_mse(W: np.ndarray):
    """
    Compute (mse_q4_0, mse_q8_0): the mean squared reconstruction error of
    W after ggml-style blockwise symmetric quantization, one block per
    row (block size = row length):
      Q4_0 -- signed 4-bit codes in [-8, 7],   scale = max(|row|) / 8
      Q8_0 -- signed 8-bit codes in [-127, 127], scale = max(|row|) / 127
    Both round-to-nearest with that per-row absmax scale.
    """
    raise NotImplementedError('your code here')
