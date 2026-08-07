import numpy as np


def gen_batch_case(seed, batch_size=4, in_dim=16, out_dim=16, rank=4, num_adapters=3):
    rng = np.random.RandomState(seed)
    x = rng.randn(batch_size, in_dim).astype(np.float32)
    adapter_ids = rng.choice(np.arange(-1, num_adapters), size=batch_size).astype(np.int32)
    lora_a = rng.randn(num_adapters, in_dim, rank).astype(np.float32)
    lora_b = rng.randn(num_adapters, rank, out_dim).astype(np.float32)
    scaling = (rng.rand(num_adapters) * 2.0 + 0.5).astype(np.float32)

    out = np.zeros((batch_size, out_dim), dtype=np.float32)
    for i in range(batch_size):
        aid = adapter_ids[i]
        if aid >= 0:
            low_rank = np.dot(x[i], lora_a[aid])
            out[i] = np.dot(low_rank, lora_b[aid]) * scaling[aid]

    return {
        "x": x,
        "adapter_ids": adapter_ids,
        "lora_a": lora_a,
        "lora_b": lora_b,
        "scaling": scaling,
        "expected_out": out,
    }


BATCH_CASES = [gen_batch_case(seed=i) for i in range(5)]


def ref_apply_per_row_lora(x, adapter_ids, lora_a, lora_b, scaling):
    batch_size, in_features = x.shape
    out_features = lora_b.shape[2]
    out = np.zeros((batch_size, out_features), dtype=x.dtype)
    for i in range(batch_size):
        aid = adapter_ids[i]
        if aid < 0:
            continue
        a = lora_a[aid]
        b = lora_b[aid]
        s = scaling[aid] if isinstance(scaling, (list, tuple, np.ndarray)) else scaling
        low_rank = np.dot(x[i], a)
        out[i] = np.dot(low_rank, b) * s
    return out


def ref_compute_adapter_memory_bytes(num_adapters, rank, num_target_layers, hidden_size, dtype_bytes=2):
    params_per_layer = 2 * rank * hidden_size
    total_params = num_adapters * num_target_layers * params_per_layer
    return int(total_params * dtype_bytes)


def ref_compute_replica_memory_bytes(num_replicas, base_params, kv_cache_bytes_per_replica, dtype_bytes=2):
    model_bytes = base_params * dtype_bytes
    per_replica = model_bytes + kv_cache_bytes_per_replica
    return int(num_replicas * per_replica)


def ref_find_adapter_vs_replica_crossover(rank, num_target_layers, hidden_size, base_params, kv_cache_bytes, max_adapters=100, dtype_bytes=2):
    target_mem = ref_compute_replica_memory_bytes(2, base_params, kv_cache_bytes, dtype_bytes)
    for n in range(1, max_adapters + 1):
        mem = ref_compute_adapter_memory_bytes(n, rank, num_target_layers, hidden_size, dtype_bytes)
        if mem >= target_mem:
            return n - 1
    return max_adapters
