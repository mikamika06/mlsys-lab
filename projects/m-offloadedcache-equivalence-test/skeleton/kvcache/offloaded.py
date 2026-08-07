import numpy as np

class DynamicCache:
    def __init__(self):
        self.key_cache = {}
        self.value_cache = {}

    def update(self, key_states, value_states, layer_idx):
        if layer_idx not in self.key_cache:
            self.key_cache[layer_idx] = key_states
            self.value_cache[layer_idx] = value_states
        else:
            self.key_cache[layer_idx] = np.concatenate(
                [self.key_cache[layer_idx], key_states], axis=-2
            )
            self.value_cache[layer_idx] = np.concatenate(
                [self.value_cache[layer_idx], value_states], axis=-2
            )
        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def get_seq_length(self, layer_idx=0):
        if layer_idx not in self.key_cache:
            return 0
        return self.key_cache[layer_idx].shape[-2]

class OffloadedCache(DynamicCache):
    def __init__(self):
        super().__init__()
        self.cpu_keys = {}
        self.cpu_values = {}

    def update(self, key_states, value_states, layer_idx):
        raise NotImplementedError

    def get_seq_length(self, layer_idx=0):
        raise NotImplementedError
