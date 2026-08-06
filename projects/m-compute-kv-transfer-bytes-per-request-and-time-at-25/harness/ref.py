import math

CONFIGS = [
    {"num_layers": 32, "num_kv_heads": 8, "head_dim": 128, "dtype_bytes": 2},
    {"num_layers": 40, "num_kv_heads": 4, "head_dim": 128, "dtype_bytes": 2},
    {"num_layers": 60, "num_kv_heads": 8, "head_dim": 64, "dtype_bytes": 2},
]

BANDWIDTHS_GBPS = [25.0, 100.0, 400.0]

REQUESTS = [
    {"prompt_len": 512, "output_len": 128},
    {"prompt_len": 2048, "output_len": 256},
    {"prompt_len": 8192, "output_len": 512},
]

def compute_kv_bytes(config, prompt_len):
    return config["num_layers"] * 2 * config["num_kv_heads"] * config["head_dim"] * config["dtype_bytes"] * prompt_len

def compute_transfer_times(kv_bytes, bandwidths_gbps):
    res = {}
    for bw in bandwidths_gbps:
        res[bw] = (kv_bytes * 8.0) / (bw * 1e9)
    return res

def compute_breakeven_prompt_len(config, bandwidth_gbps, overhead_sec=0.005):
    bpt = compute_kv_bytes(config, 1)
    bytes_per_sec = (bandwidth_gbps * 1e9) / 8.0
    min_bytes = overhead_sec * bytes_per_sec
    val = math.ceil(min_bytes / bpt) if bpt > 0 else 1
    return int(val)

def compute_sizing_ratio(prompt_len, output_len, prefill_ms_per_token=0.5, decode_ms_per_token=20.0):
    total_prefill_time = prompt_len * prefill_ms_per_token
    total_decode_time = output_len * decode_ms_per_token
    if total_decode_time == 0:
        return float('inf')
    return total_prefill_time / total_decode_time
