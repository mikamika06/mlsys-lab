import numpy as np


class DynamicCache:
    """Dynamic key-value cache supporting layer updates, cropping, and slicing."""

    def __init__(self, key_cache=None, value_cache=None):
        self.key_cache = list(key_cache) if key_cache is not None else []
        self.value_cache = list(value_cache) if value_cache is not None else []

    def update(self, key_states, value_states, layer_idx):
        while len(self.key_cache) <= layer_idx:
            self.key_cache.append(None)
            self.value_cache.append(None)

        if self.key_cache[layer_idx] is None:
            self.key_cache[layer_idx] = key_states.copy()
            self.value_cache[layer_idx] = value_states.copy()
        else:
            self.key_cache[layer_idx] = np.concatenate(
                [self.key_cache[layer_idx], key_states], axis=2
            )
            self.value_cache[layer_idx] = np.concatenate(
                [self.value_cache[layer_idx], value_states], axis=2
            )
        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def get_seq_length(self, layer_idx=0):
        if layer_idx < len(self.key_cache) and self.key_cache[layer_idx] is not None:
            return self.key_cache[layer_idx].shape[2]
        return 0

    def crop(self, max_length):
        if max_length < 0:
            raise ValueError("max_length must be non-negative")
        for i in range(len(self.key_cache)):
            if self.key_cache[i] is not None:
                cur_len = self.key_cache[i].shape[2]
                if cur_len > max_length:
                    self.key_cache[i] = self.key_cache[i][:, :, :max_length, :]
                    self.value_cache[i] = self.value_cache[i][:, :, :max_length, :]

    def slice(self, start, end):
        if start < 0 or (end is not None and end < start):
            raise ValueError("Invalid slice bounds")
        for i in range(len(self.key_cache)):
            if self.key_cache[i] is not None:
                self.key_cache[i] = self.key_cache[i][:, :, start:end, :]
                self.value_cache[i] = self.value_cache[i][:, :, start:end, :]

    def copy(self):
        new_k = [k.copy() if k is not None else None for k in self.key_cache]
        new_v = [v.copy() if v is not None else None for v in self.value_cache]
        return DynamicCache(key_cache=new_k, value_cache=new_v)
