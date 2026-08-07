def compute_memory_delta(fp16_bytes, w4a16_bytes):
    return fp16_bytes - w4a16_bytes


def compute_throughput_ratio(fp16_tps, w4a16_tps):
    if fp16_tps == 0:
        return 0.0
    return w4a16_tps / fp16_tps
