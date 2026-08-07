import numpy as np


class WindowSinkKVCache:

    def __init__(self, num_sinks: int, window_size: int, head_dim: int):
        self.num_sinks = num_sinks
        self.window_size = window_size
        self.head_dim = head_dim
        self.capacity = num_sinks + window_size
        self.k_sinks = None
        self.v_sinks = None
        self.k_window = []
        self.v_window = []
        self._total_seen = 0

    def append(self, k: np.ndarray, v: np.ndarray) -> None:
        k = np.asarray(k, dtype=np.float64)
        v = np.asarray(v, dtype=np.float64)
        n_new = k.shape[0]
        for i in range(n_new):
            ki = k[i : i + 1]
            vi = v[i : i + 1]
            if self._total_seen < self.num_sinks:
                if self.k_sinks is None:
                    self.k_sinks = ki
                    self.v_sinks = vi
                else:
                    self.k_sinks = np.concatenate([self.k_sinks, ki], axis=0)
                    self.v_sinks = np.concatenate([self.v_sinks, vi], axis=0)
            else:
                self.k_window.append(ki)
                self.v_window.append(vi)
                if len(self.k_window) > self.window_size:
                    self.k_window.pop(0)
                    self.v_window.pop(0)
            self._total_seen += 1

    def get_keys(self) -> np.ndarray:
        parts = []
        if self.k_sinks is not None and len(self.k_sinks) > 0:
            parts.append(self.k_sinks)
        if len(self.k_window) > 0:
            parts.append(np.concatenate(self.k_window, axis=0))
        if not parts:
            return np.empty((0, self.head_dim), dtype=np.float64)
        return np.concatenate(parts, axis=0)

    def get_values(self) -> np.ndarray:
        parts = []
        if self.v_sinks is not None and len(self.v_sinks) > 0:
            parts.append(self.v_sinks)
        if len(self.v_window) > 0:
            parts.append(np.concatenate(self.v_window, axis=0))
        if not parts:
            return np.empty((0, self.head_dim), dtype=np.float64)
        return np.concatenate(parts, axis=0)

    @property
    def current_seq_len(self) -> int:
        sink_len = 0 if self.k_sinks is None else len(self.k_sinks)
        return sink_len + len(self.k_window)
