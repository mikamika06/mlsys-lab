import numpy as np


def compute_adapter_memory_bytes(num_adapters, rank, num_target_layers, hidden_size, dtype_bytes=2):
    """Compute total GPU memory footprint in bytes for hosting N concurrent LoRA adapters."""
    params_per_layer = 2 * rank * hidden_size
    total_params = num_adapters * num_target_layers * params_per_layer
    return int(total_params * dtype_bytes)


def compute_replica_memory_bytes(num_replicas, base_params, kv_cache_bytes_per_replica, dtype_bytes=2):
    """Compute total GPU memory footprint in bytes for deploying N full model replicas."""
    model_bytes = base_params * dtype_bytes
    per_replica = model_bytes + kv_cache_bytes_per_replica
    return int(num_replicas * per_replica)


def find_adapter_vs_replica_crossover(rank, num_target_layers, hidden_size, base_params, kv_cache_bytes, max_adapters=100, dtype_bytes=2):
    """Find maximum number of adapters before memory equals or exceeds 2 full replicas."""
    target_mem = compute_replica_memory_bytes(2, base_params, kv_cache_bytes, dtype_bytes)
    for n in range(1, max_adapters + 1):
        mem = compute_adapter_memory_bytes(n, rank, num_target_layers, hidden_size, dtype_bytes)
        if mem >= target_mem:
            return n - 1
    return max_adapters
