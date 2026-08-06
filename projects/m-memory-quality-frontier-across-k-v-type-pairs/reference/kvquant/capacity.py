from kvquant.fallback import detect_fa_fallback

TYPE_BYTES = {
    "f32": 4.0,
    "f16": 2.0,
    "q8_0": 1.0625,
    "q4_0": 0.5625,
    "q4_1": 0.625,
}


def max_context_length(vram_budget_bytes, base_model_bytes, model_cfg, k_type="q8_0", v_type="q8_0"):
    avail = vram_budget_bytes - base_model_bytes
    if avail <= 0:
        return 0
    fb = detect_fa_fallback(k_type, v_type, model_cfg["head_dim"])
    elements_per_token = model_cfg["n_layers"] * model_cfg["n_kv_heads"] * model_cfg["head_dim"]
    unit_bytes = elements_per_token * (TYPE_BYTES[k_type] + TYPE_BYTES[v_type])
    if fb["fallback"]:
        unit_bytes *= 1.20
    return int(avail // unit_bytes)
