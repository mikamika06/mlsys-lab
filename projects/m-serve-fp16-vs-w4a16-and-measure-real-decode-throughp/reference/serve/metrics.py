def compute_throughput_ratio(fp16_tokens: float, w4a16_tokens: float) -> float:
    if fp16_tokens <= 0:
        return 0.0
    return float(w4a16_tokens / fp16_tokens)


def compute_memory_delta(fp16_bytes: int, w4a16_bytes: int) -> float:
    return float(fp16_bytes - w4a16_bytes)
