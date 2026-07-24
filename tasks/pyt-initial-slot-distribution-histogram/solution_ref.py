def initial_slot_histogram(keys, size):
    counts = [0] * size
    mask = size - 1
    for key in keys:
        counts[hash(key) & mask] += 1
    return counts
