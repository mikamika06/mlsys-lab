def kv_memory_waste_ratio(lengths, max_len, block_size):
    paged = sum(((length + block_size - 1) // block_size) * block_size for length in lengths)
    contiguous = len(lengths) * max_len
    return float(contiguous / paged)
