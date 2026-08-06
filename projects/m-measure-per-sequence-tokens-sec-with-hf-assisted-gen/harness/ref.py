def measure_tokens_per_sec(trace: list[tuple[int, int, float]], batch_size: int) -> float:
    total_accepted = sum(accepted for _, accepted, _ in trace)
    total_time = sum(latency for _, _, latency in trace)
    if total_time == 0:
        return 0.0
    return (total_accepted / batch_size) / total_time

def flops_neutral_batch_size(ar_batch_size: int, c: float, n: int) -> int:
    return int(ar_batch_size / (c * n + 1))

def find_crossover_batch_size(sweep: dict[int, tuple[float, float]]) -> int:
    crossover = 0
    for bs in sorted(sweep.keys()):
        ar_tp, spec_tp = sweep[bs]
        if spec_tp > ar_tp:
            crossover = bs
    return crossover

TRACES = [
    ([(4, 3, 0.1), (4, 4, 0.12)], 1),
    ([(4, 2, 0.1), (4, 5, 0.15)], 4),
    ([(4, 1, 0.05)], 8),
]

SWEEPS = [
    {1: (10.0, 25.0), 4: (35.0, 60.0), 8: (65.0, 70.0), 16: (120.0, 100.0)},
    {1: (15.0, 30.0), 2: (28.0, 45.0), 4: (50.0, 48.0)},
    {1: (5.0, 4.0), 2: (10.0, 8.0)}
]

FLOPS_CASES = [
    (256, 0.1, 4),
    (128, 0.2, 3),
    (64, 0.05, 5)
]
