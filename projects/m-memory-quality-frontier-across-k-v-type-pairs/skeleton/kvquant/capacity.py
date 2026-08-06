TYPE_BYTES = {
    "f32": 4.0,
    "f16": 2.0,
    "q8_0": 1.0625,
    "q4_0": 0.5625,
    "q4_1": 0.625,
}


def max_context_length(vram_budget_bytes, base_model_bytes, model_cfg, k_type="q8_0", v_type="q8_0"):
    raise NotImplementedError
