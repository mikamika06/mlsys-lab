"""KV cache budget planner and fused path analyzer."""

TYPE_BYTES_PER_ELEMENT = {
    "f32": 4.0,
    "f16": 2.0,
    "q8_0": 1.0625,
    "q4_0": 0.5625,
}


def fit_context_budget(model_config, memory_budget_bytes, kv_type_k="f16", kv_type_v="f16", block_size=32):
    num_layers = model_config["num_layers"]
    num_kv_heads = model_config["num_kv_heads"]
    head_dim = model_config["head_dim"]
    weights_bytes = model_config.get("weights_bytes", 0)
    fixed_overhead_bytes = model_config.get("fixed_overhead_bytes", 0)

    available_bytes = memory_budget_bytes - weights_bytes - fixed_overhead_bytes
    if available_bytes <= 0:
        return 0

    bytes_per_elem_k = TYPE_BYTES_PER_ELEMENT[kv_type_k.lower()]
    bytes_per_elem_v = TYPE_BYTES_PER_ELEMENT[kv_type_v.lower()]

    bytes_per_token = num_layers * num_kv_heads * head_dim * (bytes_per_elem_k + bytes_per_elem_v)
    bytes_per_block = bytes_per_token * block_size

    num_blocks = int(available_bytes // bytes_per_block)
    return num_blocks * block_size


def check_flash_attn_requirement(kv_type_k, kv_type_v, use_flash_attn):
    k_quantized = kv_type_k.lower() not in ("f16", "f32")
    v_quantized = kv_type_v.lower() not in ("f16", "f32")
    if (k_quantized or v_quantized) and not use_flash_attn:
        return False
    return True


def measure_fused_path_penalty(kv_type_k, kv_type_v):
    tk = kv_type_k.lower()
    tv = kv_type_v.lower()
    if tk == tv:
        return 1.0
    bk = TYPE_BYTES_PER_ELEMENT[tk]
    bv = TYPE_BYTES_PER_ELEMENT[tv]
    ratio = max(bk, bv) / min(bk, bv)
    return round(1.0 + 0.15 * ratio, 4)
