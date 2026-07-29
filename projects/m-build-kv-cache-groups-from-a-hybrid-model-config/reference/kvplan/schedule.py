def free_schedule(window, block_size, steps):
    out = []
    for t in range(1, steps + 1):
        oldest_kept = max(0, t - window)
        freed = oldest_kept // block_size
        out.append(freed)
    return out
