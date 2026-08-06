import numpy as np


def fp16_subnormal_mask(x: np.ndarray) -> np.ndarray:
    """Returns a boolean mask of elements that represent FP16 subnormal numbers."""
    raise NotImplementedError


def round_fp32_to_fp16(x: np.ndarray, flush_subnormals: bool = False) -> np.ndarray:
    """Casts float32 to float16, optionally flushing subnormal results to signed zero."""
    raise NotImplementedError
