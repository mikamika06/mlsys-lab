def measure_tokens_per_sec(trace: list[tuple[int, int, float]], batch_size: int) -> float:
    total_accepted = sum(accepted for _, accepted, _ in trace)
    total_time = sum(latency for _, _, latency in trace)
    if total_time == 0:
        return 0.0
    return (total_accepted / batch_size) / total_time
