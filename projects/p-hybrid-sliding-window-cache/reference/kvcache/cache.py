import numpy as np
from kvcache.config import ModelConfig


class SingleLayerCache:

    def __init__(self, is_sliding: bool, window_size: int | None):
        self.is_sliding = is_sliding
        self.window_size = window_size
        self.k: np.ndarray | None = None
        self.v: np.ndarray | None = None
        self.total_seen: int = 0

    def append(self, k: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.k is None:
            self.k = k.copy()
            self.v = v.copy()
        else:
            self.k = np.concatenate([self.k, k], axis=0)
            self.v = np.concatenate([self.v, v], axis=0)

        self.total_seen += k.shape[0]

        if self.is_sliding and self.window_size is not None:
            if self.k.shape[0] > self.window_size:
                self.k = self.k[-self.window_size :]
                self.v = self.v[-self.window_size :]

        return self.k, self.v

    def get_kv(self) -> tuple[np.ndarray | None, np.ndarray | None]:
        return self.k, self.v

    def current_slots(self) -> int:
        return 0 if self.k is None else self.k.shape[0]


class HybridKVCache:

    def __init__(self, config: ModelConfig):
        self.config = config
        self.layers = [
            SingleLayerCache(lc.is_sliding, lc.window_size)
            for lc in config.layer_configs
        ]

    def update(self, layer_idx: int, k: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return self.layers[layer_idx].append(k, v)

    def get_kv(self, layer_idx: int) -> tuple[np.ndarray | None, np.ndarray | None]:
        return self.layers[layer_idx].get_kv()

    def total_allocated_slots(self) -> int:
        return sum(layer.current_slots() for layer in self.layers)

    def naive_allocated_slots(self) -> int:
        max_seen = max((layer.total_seen for layer in self.layers), default=0)
        return max_seen * len(self.layers)

    def memory_saving_ratio(self) -> float:
        naive = self.naive_allocated_slots()
        if naive == 0:
            return 0.0
        allocated = self.total_allocated_slots()
        return (naive - allocated) / float(naive)
