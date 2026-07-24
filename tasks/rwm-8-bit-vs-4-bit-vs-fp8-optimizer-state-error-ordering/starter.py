import numpy as np


def optimizer_state_quant_compare(v: np.ndarray, block_size: int = 32) -> dict:
    """
    Quantize `v` with 8-bit blockwise, 4-bit blockwise (nibble-packed), and
    fp8-style formats; return reconstruction MSE and storage bytes for each.

    Returns a dict with keys:
      "mse_8bit", "mse_4bit", "mse_fp8", "bytes_8bit", "bytes_4bit", "bytes_fp8"
    """
    raise NotImplementedError('your code here')
