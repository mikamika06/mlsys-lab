import numpy as np


def internal_fragmentation(block_size, lengths):
    arr = np.asarray(lengths, dtype=float)
    if arr.size == 0:
        return 0.0
    remainders = arr % block_size
    wasted = np.where(remainders == 0, 0.0, block_size - remainders)
    total_allocated = np.ceil(arr / block_size) * block_size
    total_allocated = np.where(total_allocated == 0, block_size, total_allocated)
    return float(np.sum(wasted) / np.sum(total_allocated))


def table_overhead(block_size, lengths, bytes_per_entry=4):
    arr = np.asarray(lengths, dtype=float)
    if arr.size == 0:
        return 0.0
    num_blocks = np.ceil(arr / block_size)
    num_blocks = np.where(num_blocks == 0, 1.0, num_blocks)
    total_table_bytes = np.sum(num_blocks) * bytes_per_entry
    total_token_bytes = np.sum(arr) * bytes_per_entry * 2
    if total_token_bytes == 0:
        return 0.0
    return float(total_table_bytes / (np.sum(arr) * bytes_per_entry))


def total_overhead(block_size, lengths, bytes_per_entry=4):
    frag = internal_fragmentation(block_size, lengths)
    tbl = table_overhead(block_size, lengths, bytes_per_entry)
    return float(frag + tbl)
