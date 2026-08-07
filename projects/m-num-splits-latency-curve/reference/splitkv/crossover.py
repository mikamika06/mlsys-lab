from splitkv.curve import optimal_num_splits


def find_crossover_batch(seq_len: int, num_sm: int = 108) -> int:
    for b in range(1, 512):
        if optimal_num_splits(b, seq_len, num_sm) == 1:
            return b
    return 512
