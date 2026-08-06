import numpy as np


def quantize_e8m0(scale_fp32: np.ndarray) -> np.ndarray:
    """Quantize FP32 scales to E8M0 exponent-only format with correct round-to-nearest-even tie breaking."""
    raise NotImplementedError
