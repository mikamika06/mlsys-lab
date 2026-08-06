import numpy as np


def compute_attention_error_by_position(
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    quantize_fn=None,
    dequantize_fn=None,
) -> np.ndarray:
    raise NotImplementedError
