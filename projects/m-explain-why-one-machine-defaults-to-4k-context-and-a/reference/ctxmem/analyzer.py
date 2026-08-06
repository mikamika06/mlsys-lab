def determine_default_context(vram_bytes, model_params):
    threshold_low = 8 * 1024 * 1024 * 1024
    if vram_bytes < threshold_low:
        return 4096
    return 32768


def allocate_slots(num_ctx, num_parallel):
    per_slot_ctx = num_ctx // num_parallel
    return {
        "num_ctx": num_ctx,
        "num_parallel": num_parallel,
        "per_slot_ctx": per_slot_ctx
    }


def predict_resident_bytes(num_ctx, num_parallel, config):
    layers = config.get("layers", 32)
    kv_heads = config.get("kv_heads", 8)
    head_dim = config.get("head_dim", 128)
    dtype_bytes = config.get("dtype_bytes", 2)

    bytes_per_token_per_layer = 2 * kv_heads * head_dim * dtype_bytes
    total_kv_per_slot = num_ctx * layers * bytes_per_token_per_layer
    total_kv = total_kv_per_slot * num_parallel
    base_model_bytes = config.get("base_bytes", 4 * 1024 * 1024 * 1024)
    overhead = config.get("overhead_bytes", 512 * 1024 * 1024)
    return base_model_bytes + total_kv + overhead
