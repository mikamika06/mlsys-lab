import math


def predict_latency(batch_size: int, seq_len: int, num_splits: int, num_sm: int = 108) -> float:
    total_ctas = batch_size * num_splits
    waves = math.ceil(total_ctas / float(num_sm))
    tokens_per_split = math.ceil(seq_len / float(num_splits))
    compute_cost = 0.05 * tokens_per_split
    reduction_overhead = 0.0 if num_splits == 1 else (12.0 + 3.5 * num_splits)
    return float(waves * (compute_cost + reduction_overhead))


def optimal_num_splits(batch_size: int, seq_len: int, num_sm: int = 108) -> int:
    best_s = 1
    best_lat = float("inf")
    for s in range(1, 65):
        lat = predict_latency(batch_size, seq_len, s, num_sm)
        if lat < best_lat:
            best_lat = lat
            best_s = s
    return best_s
