def max_throughput_batch_size(
    max_batch: int,
    slo_ms: float,
    fixed_ms: float,
    per_item_ms: float,
    jitter_ms: float,
) -> int:
    best_b = 1
    best_t = -1.0
    found = False
    for b in range(1, max_batch + 1):
        latency = fixed_ms + per_item_ms * b + jitter_ms * (b ** 2)
        if latency <= slo_ms:
            found = True
            throughput = 1000.0 * b / latency
            if throughput > best_t:
                best_t = throughput
                best_b = b
    return best_b if found else 1
