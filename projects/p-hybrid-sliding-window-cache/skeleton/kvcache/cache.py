import numpy as np
from kvcache.config import ModelConfig


class SingleLayerCache:

    def __init__(self, is_sliding: bool, window_size: int | None):
        raise NotImplementedError

    def append(self, k: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError

    def get_kv(self) -> tuple[np.ndarray | None, np.ndarray | None]:
        raise NotImplementedError

    def current_slots(self) -> int:
        raise NotImplementedError


class HybridKVCache:

    def __init__(self, config: ModelConfig):
        raise NotImplementedError

    def update(self, layer_idx: int, k: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError

    def get_kv(self, layer_idx: int) -> tuple[np.ndarray | None, np.ndarray | None]:
        raise NotImplementedError

    def total_allocated_slots(self) -> int:
        raise NotImplementedError

    def naive_allocated_slots(self) -> int:
        raise NotImplementedError

    def memory_saving_ratio(self) -> float:
        raise NotImplementedError
