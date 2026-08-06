CONFIGS = [
    (
        {
            "num_layers": 32,
            "hidden_size": 4096,
            "intermediate_size": 11008,
            "vocab_size": 32000,
            "num_kv_heads": 32,
            "head_dim": 128,
            "bytes_per_param": 2,
            "bytes_per_elem": 2
        },
        {
            "num_layers": 8,
            "hidden_size": 4096,
            "intermediate_size": 11008,
            "vocab_size": 32000,
            "num_kv_heads": 32,
            "head_dim": 128,
            "bytes_per_param": 2,
            "bytes_per_elem": 2,
            "is_self_speculative": True
        },
        4,
        1024
    ),
    (
        {
            "num_layers": 24,
            "hidden_size": 2048,
            "intermediate_size": 5600,
            "vocab_size": 32000,
            "num_kv_heads": 16,
            "head_dim": 128,
            "bytes_per_param": 2,
            "bytes_per_elem": 2
        },
        {
            "num_layers": 6,
            "hidden_size": 2048,
            "intermediate_size": 5600,
            "vocab_size": 32000,
            "num_kv_heads": 16,
            "head_dim": 128,
            "bytes_per_param": 2,
            "bytes_per_elem": 2,
            "is_self_speculative": True
        },
        2,
        512
    ),
    (
        {
            "num_layers": 16,
            "hidden_size": 1024,
            "intermediate_size": 2800,
            "vocab_size": 16000,
            "num_kv_heads": 8,
            "head_dim": 128,
            "bytes_per_param": 2,
            "bytes_per_elem": 2
        },
        {
            "num_layers": 16,
            "hidden_size": 1024,
            "intermediate_size": 2800,
            "vocab_size": 16000,
            "num_kv_heads": 8,
            "head_dim": 128,
            "bytes_per_param": 2,
            "bytes_per_elem": 2,
            "is_self_speculative": False
        },
        1,
        256
    )
]


def compute_weight_memory(config):
    num_layers = config["num_layers"]
    hidden_size = config["hidden_size"]
    intermediate_size = config["intermediate_size"]
    vocab_size = config["vocab_size"]
    bytes_per_param = config.get("bytes_per_param", 2)
    attn_weights = 4 * (hidden_size * hidden_size)
    mlp_weights = 3 * (hidden_size * intermediate_size)
    layer_weights = attn_weights + mlp_weights
    total = (num_layers * layer_weights) + (vocab_size * hidden_size)
    return total * bytes_per_param


def compute_kv_cache_memory(config, batch_size, seq_len):
    num_layers = config["num_layers"]
    num_kv_heads = config["num_kv_heads"]
    head_dim = config["head_dim"]
    bytes_per_elem = config.get("bytes_per_elem", 2)
    cache_per_token = 2 * num_layers * num_kv_heads * head_dim * bytes_per_elem
    return batch_size * seq_len * cache_per_token


def derive_savings(target_config, draft_config, batch_size, seq_len):
    target_weights = compute_weight_memory(target_config)
    target_kv = compute_kv_cache_memory(target_config, batch_size, seq_len)
    target_total = target_weights + target_kv
    is_self_spec = draft_config.get("is_self_speculative", False)
    if is_self_spec:
        draft_weights = target_weights
    else:
        draft_weights = compute_weight_memory(draft_config)
    draft_kv = compute_kv_cache_memory(draft_config, batch_size, seq_len)
    draft_total = draft_weights + draft_kv
    combined_total = target_total if is_self_spec else (target_total + draft_total)
    separate_total = target_total + draft_total
    saved_bytes = separate_total - combined_total
    savings_ratio = saved_bytes / separate_total if separate_total > 0 else 0.0
    return {
        "target_total": target_total,
        "draft_total": draft_total,
        "combined_total": combined_total,
        "saved_bytes": saved_bytes,
        "savings_ratio": savings_ratio
    }
