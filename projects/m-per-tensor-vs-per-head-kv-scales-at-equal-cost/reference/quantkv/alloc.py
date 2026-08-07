def allocate_bits(total_budget_bytes: int, num_weights: int, num_kv_elements: int, target_bits_w: int) -> dict:
    """Allocate bits between weights and KV cache under budget."""
    w_bytes = (num_weights * target_bits_w) // 8
    remaining = total_budget_bytes - w_bytes
    if remaining < 0:
        return {"bits_w": target_bits_w, "bits_kv": 0, "feasible": False}
    kv_bits_per_elem = (remaining * 8) / max(1, num_kv_elements)
    if kv_bits_per_elem >= 8:
        bits_kv = 8
    elif kv_bits_per_elem >= 4:
        bits_kv = 4
    elif kv_bits_per_elem >= 2:
        bits_kv = 2
    else:
        bits_kv = 0
    return {"bits_w": target_bits_w, "bits_kv": bits_kv, "feasible": True}
