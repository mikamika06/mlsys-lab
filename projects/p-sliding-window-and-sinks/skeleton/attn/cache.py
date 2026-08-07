import numpy as np


class WindowSinkKVCache:

    def __init__(self, num_sinks: int, window_size: int, head_dim: int):
        raise NotImplementedError

    def append(self, k: np.ndarray, v: np.ndarray) -> None:
        raise NotImplementedError

    def get_keys(self) -> np.ndarray:
        raise NotImplementedError

    def get_values(self) -> np.ndarray:
        raise NotImplementedError

    @property
    def current_seq_len(self) -> int:
        raise NotImplementedError

    @property
    def capacity(self) -> int:
        raise NotImplementedError
