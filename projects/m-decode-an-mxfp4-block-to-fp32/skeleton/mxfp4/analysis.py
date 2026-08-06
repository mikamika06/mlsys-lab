import numpy as np


def enumerate_mxfp4_grid(scale_e8m0: int) -> np.ndarray:
    """Return all 16 unique FP32 values representable by MXFP4 for a given scale_e8m0."""
    raise NotImplementedError


def mxfp4_vs_q4_0_crossover(blocks: np.ndarray) -> dict[str, float]:
    """Compare reconstruction MSE between MXFP4 and Q4_0 across input blocks."""
    raise NotImplementedError
