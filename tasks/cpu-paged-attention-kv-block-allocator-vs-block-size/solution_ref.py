import math

def choose_kv_block_size(token_sizes, table_overhead_per_block):
    best_b = None
    best_ratio = -1.0
    total_useful = sum(token_sizes)
    candidates = [16, 32, 64, 128, 256, 512, 1024]
    for b in candidates:
        total_blocks = 0
        total_alloc = 0
        for s in token_sizes:
            blocks = math.ceil(s / b)
            total_blocks += blocks
            total_alloc += blocks * b
        total_alloc += total_blocks * table_overhead_per_block
        ratio = total_useful / total_alloc
        if ratio > best_ratio + 1e-12 or (abs(ratio - best_ratio) < 1e-12 and (best_b is None or b < best_b)):
            best_ratio = ratio
            best_b = b
    return best_b, best_ratio
