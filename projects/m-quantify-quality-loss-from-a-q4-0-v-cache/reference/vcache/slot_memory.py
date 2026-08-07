import numpy as np


def compute_slot_kv_bytes(n_layers, n_kv_heads, head_dim, seq_len, n_parallel, element_size_k=2, v_type="q4_0", block_size=32):
    """Compute total bytes required for K and V cache with parallel slots."""
    if v_type == "f16":
        v_bytes_per_elem = 2.0
    elif v_type == "q4_0":
        v_bytes_per_elem = (2.0 + block_size // 2) / block_size
    else:
        v_bytes_per_elem = 2.0

    k_bytes_per_token = n_layers * n_kv_heads * head_dim * element_size_k
    v_bytes_per_token = int(np.ceil(n_layers * n_kv_heads * head_dim * v_bytes_per_elem))

    single_slot_bytes = (k_bytes_per_token + v_bytes_per_token) * seq_len
    total_bytes = single_slot_bytes * n_parallel
    return int(total_bytes)


def predict_multi_slot_growth(configs):
    """Predict KV cache byte sizes for multiple slot configurations."""
    res = []
    for cfg in configs:
        b = compute_slot_kv_bytes(
            cfg["n_layers"],
            cfg["n_kv_heads"],
            cfg["head_dim"],
            cfg["seq_len"],
            cfg["n_parallel"],
            cfg.get("element_size_k", 2),
            cfg.get("v_type", "q4_0"),
            cfg.get("block_size", 32),
        )
        res.append(b)
    return res
