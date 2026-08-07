"""Offloaded KV Cache implementation."""

import torch


class OffloadedCache:

    def __init__(self, num_layers, offload_device="cpu"):
        self.num_layers = num_layers
        self.offload_device = offload_device
        self.key_cache = [None] * num_layers
        self.value_cache = [None] * num_layers

    def update(self, key_states, value_states, layer_idx):
        if self.key_cache[layer_idx] is None:
            self.key_cache[layer_idx] = key_states
            self.value_cache[layer_idx] = value_states
        else:
            self.key_cache[layer_idx] = torch.cat(
                [self.key_cache[layer_idx], key_states], dim=-2
            )
            self.value_cache[layer_idx] = torch.cat(
                [self.value_cache[layer_idx], value_states], dim=-2
            )
        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def get_seq_length(self, layer_idx=0):
        if self.key_cache[layer_idx] is None:
            return 0
        return self.key_cache[layer_idx].shape[-2]

    def evict_to_offload(self, layer_idx):
        if self.key_cache[layer_idx] is not None:
            self.key_cache[layer_idx] = self.key_cache[layer_idx].to(
                self.offload_device
            )
            self.value_cache[layer_idx] = self.value_cache[layer_idx].to(
                self.offload_device
            )

    def prefetch_to_device(self, layer_idx, target_device):
        if self.key_cache[layer_idx] is not None:
            self.key_cache[layer_idx] = self.key_cache[layer_idx].to(
                target_device
            )
            self.value_cache[layer_idx] = self.value_cache[layer_idx].to(
                target_device
            )
