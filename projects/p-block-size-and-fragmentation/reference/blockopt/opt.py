import numpy as np
from blockopt.models import total_overhead


def find_optimal_block_size(lengths, block_sizes, bytes_per_entry=4):
    best_bs = block_sizes[0]
    best_val = float("inf")
    for bs in block_sizes:
        val = total_overhead(bs, lengths, bytes_per_entry)
        if val < best_val:
            best_val = val
            bs_best = bs
        elif val == best_val:
            if bs < bs_best:
                bs_best = bs
    return int(bs_best)


def evaluate_trace(block_size, trace):
    lengths = [t["len"] for t in trace]
    return total_overhead(block_size, lengths)


def check_threshold(block_size, lengths, max_loss_ratio):
    loss = total_overhead(block_size, lengths)
    return bool(loss <= max_loss_ratio)


def recommend_block_size(lengths_distribution):
    mean_len = float(np.mean(lengths_distribution))
    if mean_len < 32:
        return 16
    elif mean_len < 128:
        return 64
    else:
        return 256
