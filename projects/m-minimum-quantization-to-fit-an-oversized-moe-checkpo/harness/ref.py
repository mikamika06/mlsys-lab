MODELS = [
    {
        "name": "MoE-8x7B",
        "total_params": 47e9,
        "expert_params": 13e9,
        "shared_params": 5e9,
        "layers": 32,
    },
    {
        "name": "MoE-8x22B",
        "total_params": 141e9,
        "expert_params": 39e9,
        "shared_params": 15e9,
        "layers": 56,
    },
    {
        "name": "MoE-16x12B",
        "total_params": 120e9,
        "expert_params": 30e9,
        "shared_params": 10e9,
        "layers": 40,
    },
]

def compute_size_bytes(spec, bits, overhead_factor=1.05):
    params = spec["total_params"]
    bytes_per_param = bits / 8.0
    return int(params * bytes_per_param * overhead_factor)

def min_bits(spec, limit_bytes=36 * 1024 * 1024 * 1024):
    allowed_bits = [2, 3, 4, 5, 6, 8, 16]
    for b in allowed_bits:
        if compute_size_bytes(spec, b) <= limit_bytes:
            return b
    return 16

def estimate_mlx_4bit(spec):
    base = compute_size_bytes(spec, 4, overhead_factor=1.02)
    return {"bytes": base, "bits_per_weight": 4.15}

def estimate_gguf_q4km(spec):
    base = compute_size_bytes(spec, 4, overhead_factor=1.08)
    return {"bytes": base, "bits_per_weight": 4.58}

def compare_formats(spec):
    mlx = estimate_mlx_4bit(spec)
    gguf = estimate_gguf_q4km(spec)
    return {"mlx": mlx, "gguf": gguf}
