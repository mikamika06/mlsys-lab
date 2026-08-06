CONFIGS = [
    {"head_dim": 64, "seq_len": 2048, "num_heads": 32, "dtype": "fp16", "hardware": "hopper"},
    {"head_dim": 128, "seq_len": 4096, "num_heads": 16, "dtype": "fp16", "hardware": "hopper"},
    {"head_dim": 256, "seq_len": 8192, "num_heads": 8, "dtype": "fp16", "hardware": "hopper"},
    {"head_dim": 192, "seq_len": 2048, "num_heads": 16, "dtype": "bf16", "hardware": "hopper"},
    {"head_dim": 128, "seq_len": 4096, "num_heads": 16, "dtype": "fp8", "hardware": "hopper"},
]


def check_head_ceiling(cfg):
    hd = cfg["head_dim"]
    if hd <= 64:
        return {"max_supported_dim": 64, "fa2_supported": True, "fa3_supported": True, "optimal_block": 64}
    elif hd <= 128:
        return {"max_supported_dim": 128, "fa2_supported": True, "fa3_supported": True, "optimal_block": 128}
    elif hd <= 256:
        return {"max_supported_dim": 256, "fa2_supported": True, "fa3_supported": False, "optimal_block": 128}
    else:
        return {"max_supported_dim": 512, "fa2_supported": False, "fa3_supported": False, "optimal_block": 256}


def estimate_throughput(cfg):
    hd = cfg["head_dim"]
    hw = cfg["hardware"]
    if hw != "hopper":
        return {"fa2": 50.0, "fa3": 40.0}
    if hd <= 128:
        return {"fa2": 220.0, "fa3": 310.0}
    elif hd <= 256:
        return {"fa2": 150.0, "fa3": 110.0}
    else:
        return {"fa2": 80.0, "fa3": 60.0}


def check_fp8_availability(cfg):
    hd = cfg["head_dim"]
    dt = cfg["dtype"]
    if dt != "fp8":
        return {"available": True, "reason": "not_fp8"}
    if hd % 16 != 0:
        return {"available": False, "reason": "alignment_not_multiple_of_16"}
    if hd > 128:
        return {"available": False, "reason": "head_dim_exceeds_fp8_limit"}
    return {"available": True, "reason": "supported"}
