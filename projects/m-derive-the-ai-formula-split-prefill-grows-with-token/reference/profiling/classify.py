from profiling.derivation import derive_ai_formula

def classify_phase_bound(config, phase, tokens, peak_flops, peak_bandwidth_gbps):
    derived = derive_ai_formula(
        config["params"],
        config["hidden_size"],
        config["num_layers"],
        config["total_weight_bytes"],
        tokens if phase == "prefill" else 1
    )
    if phase == "prefill":
        flops = derived["prefill_flops_per_token"] * tokens
        bytes_moved = derived["prefill_bytes_per_token"] * tokens
        time_s = config["time_prefill_ms"] / 1000.0
    else:
        flops = derived["decode_flops_per_token"] * tokens
        bytes_moved = derived["decode_bytes_per_token"] * tokens
        time_s = (config["time_decode_ms_per_token"] * tokens) / 1000.0
    intensity = flops / bytes_moved if bytes_moved > 0 else 0.0
    ridge_intensity = peak_flops / (peak_bandwidth_gbps * 1e9)
    bound = "memory" if intensity < ridge_intensity else "compute"
    achieved_bw = bytes_moved / time_s if time_s > 0 else 0.0
    return {
        "intensity": float(intensity),
        "bound": str(bound),
        "achieved_bandwidth": float(achieved_bw)
    }
