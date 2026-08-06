from speculative.memory import compute_kv_cache_memory, compute_weight_memory


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
