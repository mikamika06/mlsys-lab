"""Offloaded KV Cache implementation."""


class OffloadedCache:

    def __init__(self, num_layers, offload_device="cpu"):
        raise NotImplementedError

    def update(self, key_states, value_states, layer_idx):
        raise NotImplementedError

    def get_seq_length(self, layer_idx=0):
        raise NotImplementedError

    def evict_to_offload(self, layer_idx):
        raise NotImplementedError

    def prefetch_to_device(self, layer_idx, target_device):
        raise NotImplementedError
