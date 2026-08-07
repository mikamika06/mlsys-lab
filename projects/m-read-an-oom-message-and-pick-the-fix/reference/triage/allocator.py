def simulate_allocator(trace, max_split_size_mb):
    limit_bytes = max_split_size_mb * 1024 * 1024 if max_split_size_mb else float("inf")
    active_blocks = {}
    free_blocks = [(0, 1024 * 1024 * 1024)]
    fragmentation_score = 0

    for op_id, size in trace:
        if size > 0:
            allocated = False
            for idx, (start, length) in enumerate(free_blocks):
                if length >= size:
                    free_blocks.pop(idx)
                    if length > size and length - size <= limit_bytes:
                        free_blocks.append((start + size, length - size))
                    active_blocks[op_id] = (start, size)
                    allocated = True
                    break
            if not allocated:
                fragmentation_score += 1
        else:
            target_id = -size
            if target_id in active_blocks:
                start, size = active_blocks.pop(target_id)
                free_blocks.append((start, size))
    return fragmentation_score


def tune_max_split_size(trace: list, candidate_sizes: list) -> int:
    best_size = candidate_sizes[0]
    min_score = float("inf")
    for sz in candidate_sizes:
        score = simulate_allocator(trace, sz)
        if score < min_score:
            min_score = score
            best_size = sz
    return best_size
