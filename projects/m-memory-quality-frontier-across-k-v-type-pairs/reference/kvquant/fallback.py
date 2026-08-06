def detect_fa_fallback(k_type, v_type, head_dim):
    supported = {"f16", "q8_0", "q4_0"}
    if k_type not in supported or v_type not in supported:
        return {"fallback": True, "reason": "unsupported_quant_type"}
    if k_type != v_type:
        return {"fallback": True, "reason": "mismatch_kv_types"}
    if head_dim <= 0 or head_dim > 256 or head_dim % 32 != 0:
        return {"fallback": True, "reason": "unaligned_head_dim"}
    return {"fallback": False, "reason": ""}
