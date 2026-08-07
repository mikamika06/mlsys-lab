import numpy as np


def generate_sliding_window_mask(
    q_len: int,
    kv_len: int,
    window_size: int,
    num_sinks: int = 0,
    is_causal: bool = True,
) -> np.ndarray:
    q_idx = np.arange(q_len)[:, None]
    k_idx = np.arange(kv_len)[None, :]
    offset = kv_len - q_len

    mask = np.ones((q_len, kv_len), dtype=bool)

    if is_causal:
        mask = mask & (k_idx <= (q_idx + offset))

    window_mask = k_idx >= (q_idx + offset - window_size + 1)
    if num_sinks > 0:
        sink_mask = k_idx < num_sinks
        window_mask = window_mask | sink_mask

    mask = mask & window_mask
    return mask
