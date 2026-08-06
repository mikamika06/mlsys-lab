def find_crossover_batch_size(sweep: dict[int, tuple[float, float]]) -> int:
    crossover = 0
    for bs in sorted(sweep.keys()):
        ar_tp, spec_tp = sweep[bs]
        if spec_tp > ar_tp:
            crossover = bs
    return crossover
