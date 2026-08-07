import math

def internal_fragmentation(lengths, block_size):
    total_wasted = 0
    for l in lengths:
        remainder = l % block_size
        if remainder > 0:
            total_wasted += (block_size - remainder)
    return total_wasted

def block_table_overhead(lengths, block_size, pointer_size_bytes=8):
    total_pointers = 0
    for l in lengths:
        num_blocks = math.ceil(l / block_size)
        total_pointers += num_blocks
    return total_pointers * pointer_size_bytes

def total_memory_overhead(lengths, block_size, pointer_size_bytes=8, bytes_per_token=0):
    frag = internal_fragmentation(lengths, block_size) * bytes_per_token
    overhead = block_table_overhead(lengths, block_size, pointer_size_bytes)
    return frag + overhead

def find_optimal_block_size(lengths, block_sizes, pointer_size_bytes=8, bytes_per_token=1):
    best_size = block_sizes[0]
    min_cost = float("inf")
    for bs in block_sizes:
        cost = total_memory_overhead(lengths, bs, pointer_size_bytes, bytes_per_token)
        if cost < min_cost:
            min_cost = cost
            best_size = bs
    return best_size

def verify_trace_simulation(trace, block_size):
    lengths = [r["len"] for r in trace]
    frag = internal_fragmentation(lengths, block_size)
    tbl = block_table_overhead(lengths, block_size)
    return {"fragmentation": frag, "table_overhead": tbl}

def check_memory_threshold(lengths, block_size, max_loss_ratio):
    total_tokens = sum(lengths)
    if total_tokens == 0:
        return True
    wasted = internal_fragmentation(lengths, block_size)
    ratio = wasted / total_tokens
    return ratio <= max_loss_ratio

def recommend_block_size(trace, block_sizes, pointer_size_bytes=8, bytes_per_token=1, max_loss_ratio=0.5):
    lengths = [r["len"] for r in trace]
    return find_optimal_block_size(lengths, block_sizes, pointer_size_bytes, bytes_per_token)
