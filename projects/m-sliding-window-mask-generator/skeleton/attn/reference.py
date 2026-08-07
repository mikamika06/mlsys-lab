import numpy as np


def windowed_attention_reference(
    query: np.ndarray,
    key: np.ndarray,
    value: np.ndarray,
    window_size: int,
    num_sinks: int = 0,
    is_causal: bool = True,
) -> np.ndarray:
    """Compute reference scaled dot-product attention with windowed mask."""
    raise NotImplementedError
