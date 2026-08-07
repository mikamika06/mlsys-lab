import numpy as np


def generate_sliding_window_mask(
    q_len: int,
    kv_len: int,
    window_size: int,
    num_sinks: int = 0,
    is_causal: bool = True,
) -> np.ndarray:
    """Generate a boolean mask where True indicates allowed attention positions."""
    raise NotImplementedError
