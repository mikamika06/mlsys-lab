CONFIGS = [
    {"base_bytes": 2048, "bytes_per_token": 32},
    {"base_bytes": 4096, "bytes_per_token": 64},
    {"base_bytes": 1024, "bytes_per_token": 16},
    {"base_bytes": 8192, "bytes_per_token": 128},
]


def find_largest_context(config, budget_bytes):
    low = 1
    high = 131072
    best = 0
    while low <= high:
        mid = (low + high) // 2
        cost = config["base_bytes"] + mid * config["bytes_per_token"]
        if cost <= budget_bytes:
            best = mid
            low = mid + 1
        else:
            high = mid - 1
    return best


def requires_flash_attention(quant_type):
    qt = quant_type.lower()
    return "q4" in qt or "q8" in qt or "iq" in qt


def compute_fused_penalty(k_type, v_type):
    if k_type != v_type:
        return 1.35
    return 1.0
