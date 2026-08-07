import numpy as np
from attn.cache import WindowSinkKVCache


def compute_window_sink_attention(
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    num_sinks: int,
    window_size: int,
) -> np.ndarray:
    raise NotImplementedError


class StreamingAttentionRunner:

    def __init__(self, num_sinks: int, window_size: int, head_dim: int):
        raise NotImplementedError

    def step(self, q_tok: np.ndarray, k_tok: np.ndarray, v_tok: np.ndarray) -> np.ndarray:
        raise NotImplementedError
