import numpy as np


def quant_granularity_errors(W: np.ndarray) -> dict:
    """Quantize W with symmetric INT8 both per-tensor (one scale for the
    whole matrix) and per-channel (one scale per row), report each
    reconstruction's MSE, and pick the winner (the scheme with the lower
    MSE).

    W: (rows, cols) float64 weight matrix.

    Returns a dict:
        {
            "mse_per_tensor": float,
            "mse_per_channel": float,
            "winner": "per_tensor" | "per_channel",
        }
    """
    raise NotImplementedError('your code here')
