from specmem.model import compute_parameter_bytes, compute_kv_cache_bytes


def separate_draft_footprint(target_config, draft_config, batch_size, seq_len):
    target_params = compute_parameter_bytes(target_config)
    draft_params = compute_parameter_bytes(draft_config)
    target_kv = compute_kv_cache_bytes(target_config, batch_size, seq_len)
    draft_kv = compute_kv_cache_bytes(draft_config, batch_size, seq_len)
    return {
        "param_bytes": target_params + draft_params,
        "kv_bytes": target_kv + draft_kv,
        "total_bytes": target_params + draft_params + target_kv + draft_kv
    }


def self_speculative_footprint(target_config, extra_config, batch_size, seq_len):
    target_params = compute_parameter_bytes(target_config)
    extra_params = compute_parameter_bytes(extra_config)
    target_kv = compute_kv_cache_bytes(target_config, batch_size, seq_len)
    return {
        "param_bytes": target_params + extra_params,
        "kv_bytes": target_kv,
        "total_bytes": target_params + extra_params + target_kv
    }
