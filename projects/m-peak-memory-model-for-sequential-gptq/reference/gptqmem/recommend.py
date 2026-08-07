def recommend_fix_from_timeline(timeline):
    """Analyzes memory timeline step entries and returns targeted optimization fix."""
    peak_entry = max(timeline, key=lambda x: x["vram_bytes"])
    phase = peak_entry["phase"]

    if phase == "hessian_inversion":
        return "enable_dampening_and_fp32_cpu_inversion"
    elif phase == "activation_caching":
        return "offload_activations_to_disk"
    elif phase == "weight_quant_loop":
        return "reduce_block_size_or_disable_act_order"
    elif phase == "layer_loading":
        return "enable_sequential_layer_offloading"
    return "reduce_calibration_samples"
