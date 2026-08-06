import numpy as np

def get_test_cases():
    np.random.seed(42)
    cases = []
    for i in range(5):
        params = 7e9 + i * 1e9
        hidden_size = 4096 + i * 512
        num_layers = 32 + i * 4
        bytes_per_param = 2
        total_weight_bytes = int(params * bytes_per_param)
        prompt_tokens = 128 * (i + 1)
        gen_tokens = 64 * (i + 1)
        time_prefill_ms = 10.0 + i * 5.0
        time_decode_ms_per_token = 15.0 + i * 2.0
        peak_bandwidth_gbps = 150.0 + i * 10.0
        peak_flops = 1e12 + i * 1e11

        cases.append({
            "params": params,
            "hidden_size": hidden_size,
            "num_layers": num_layers,
            "total_weight_bytes": total_weight_bytes,
            "prompt_tokens": prompt_tokens,
            "gen_tokens": gen_tokens,
            "time_prefill_ms": time_prefill_ms,
            "time_decode_ms_per_token": time_decode_ms_per_token,
            "peak_bandwidth_gbps": peak_bandwidth_gbps,
            "peak_flops": peak_flops
        })
    return cases

def derive_formula(config):
    p = config["params"]
    prefill_flops_per_token = 2.0 * p
    decode_flops_per_token = 2.0 * p
    weight_bytes = config["total_weight_bytes"]
    prefill_bytes_per_token = weight_bytes / config["prompt_tokens"] + 2 * config["hidden_size"] * config["num_layers"]
    decode_bytes_per_token = weight_bytes + 2 * config["hidden_size"] * config["num_layers"] * 2
    return {
        "prefill_flops_per_token": prefill_flops_per_token,
        "decode_flops_per_token": decode_flops_per_token,
        "prefill_bytes_per_token": prefill_bytes_per_token,
        "decode_bytes_per_token": decode_bytes_per_token
    }

def classify_phase(config, phase, tokens=128):
    ref_vals = derive_formula(config)
    if phase == "prefill":
        flops = ref_vals["prefill_flops_per_token"] * tokens
        bytes_moved = ref_vals["prefill_bytes_per_token"] * tokens
        time_s = config["time_prefill_ms"] / 1000.0
    else:
        flops = ref_vals["decode_flops_per_token"] * tokens
        bytes_moved = ref_vals["decode_bytes_per_token"] * tokens
        time_s = (config["time_decode_ms_per_token"] * tokens) / 1000.0

    intensity = flops / bytes_moved if bytes_moved > 0 else 0
    ridge_intensity = config["peak_flops"] / (config["peak_bandwidth_gbps"] * 1e9)
    bound = "memory" if intensity < ridge_intensity else "compute"
    achieved_bandwidth = bytes_moved / time_s if time_s > 0 else 0
    return {
        "intensity": intensity,
        "bound": bound,
        "achieved_bandwidth": achieved_bandwidth
    }

def measure_bandwidth(config):
    weight_bytes = config["total_weight_bytes"]
    time_s = config["time_decode_ms_per_token"] / 1000.0
    bandwidth_gbps = (weight_bytes / time_s) / 1e9
    return bandwidth_gbps
