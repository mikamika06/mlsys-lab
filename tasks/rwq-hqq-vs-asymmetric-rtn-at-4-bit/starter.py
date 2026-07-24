import numpy as np


def compare_4bit_quantizers(x: np.ndarray) -> tuple[float, float]:
    """Compare HQQ and asymmetric RTN 4-bit reconstruction MSE."""
    x = np.asarray(x, dtype=np.float64)
    raise NotImplementedError
