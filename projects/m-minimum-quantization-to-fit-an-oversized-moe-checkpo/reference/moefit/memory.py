def calculate_memory_bytes(spec, bits, overhead_factor=1.05):
    params = spec["total_params"]
    bytes_per_param = bits / 8.0
    return int(params * bytes_per_param * overhead_factor)

def estimate_mlx_4bit(spec):
    base = calculate_memory_bytes(spec, 4, overhead_factor=1.02)
    return {"bytes": base, "bits_per_weight": 4.15}

def estimate_gguf_q4km(spec):
    base = calculate_memory_bytes(spec, 4, overhead_factor=1.08)
    return {"bytes": base, "bits_per_weight": 4.58}
