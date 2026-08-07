def fit_context_budget(model_config, memory_budget_bytes, kv_type_k="f16", kv_type_v="f16", block_size=32):
    raise NotImplementedError


def check_flash_attn_requirement(kv_type_k, kv_type_v, use_flash_attn):
    raise NotImplementedError


def measure_fused_path_penalty(kv_type_k, kv_type_v):
    raise NotImplementedError
