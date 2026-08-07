import numpy as np


def compute_adapter_memory_bytes(num_adapters, rank, num_target_layers, hidden_size, dtype_bytes=2):
    """Compute total GPU memory footprint in bytes for hosting N concurrent LoRA adapters."""
    raise NotImplementedError


def compute_replica_memory_bytes(num_replicas, base_params, kv_cache_bytes_per_replica, dtype_bytes=2):
    """Compute total GPU memory footprint in bytes for deploying N full model replicas."""
    raise NotImplementedError


def find_adapter_vs_replica_crossover(rank, num_target_layers, hidden_size, base_params, kv_cache_bytes, max_adapters=100, dtype_bytes=2):
    """Find maximum number of adapters before memory equals or exceeds 2 full replicas."""
    raise NotImplementedError
