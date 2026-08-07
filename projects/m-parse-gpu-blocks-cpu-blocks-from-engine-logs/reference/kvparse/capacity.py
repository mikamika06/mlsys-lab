from kvparse.parser import parse_blocks


def analyze_tp_scaling(log_tp1: str, log_tp2: str) -> dict:
    b1 = parse_blocks(log_tp1)
    b2 = parse_blocks(log_tp2)
    gpu1 = b1["gpu_blocks"]
    gpu2 = b2["gpu_blocks"]
    ratio = gpu2 / float(gpu1) if gpu1 > 0 else 0.0
    doubles = ratio >= 1.9
    return {
        "gpu_blocks_tp1": gpu1,
        "gpu_blocks_tp2": gpu2,
        "scaling_ratio": ratio,
        "doubles_capacity": doubles
    }
