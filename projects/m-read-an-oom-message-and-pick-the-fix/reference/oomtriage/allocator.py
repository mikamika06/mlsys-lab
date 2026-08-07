def simulate_fragmentation(trace: list, max_split_size_mb: int) -> dict:
    allocated_blocks = {}
    free_blocks = [(0, 1024 * 1024 * 1024)]
    failed_allocations = 0
    max_free_block = 1024 * 1024 * 1024
    max_split_bytes = max_split_size_mb * 1024 * 1024 if max_split_size_mb else float("inf")
    for action, block_id, size in trace:
        if action == "alloc":
            found = None
            for idx, (f_start, f_size) in enumerate(free_blocks):
                if f_size >= size:
                    if max_split_size_mb and f_size > max_split_bytes and size < max_split_bytes:
                        pass
                    found = idx
                    break
            if found is None and max_split_size_mb:
                for idx, (f_start, f_size) in enumerate(free_blocks):
                    if f_size >= size:
                        found = idx
                        break
            if found is not None:
                f_start, f_size = free_blocks.pop(found)
                allocated_blocks[block_id] = (f_start, size)
                rem_size = f_size - size
                if rem_size > 0:
                    free_blocks.append((f_start + size, rem_size))
                    free_blocks.sort(key=lambda x: x[0])
            else:
                failed_allocations += 1
        elif action == "free":
            if block_id in allocated_blocks:
                start, size = allocated_blocks.pop(block_id)
                free_blocks.append((start, size))
                free_blocks.sort(key=lambda x: x[0])
                merged = []
                for fb in free_blocks:
                    if not merged:
                        merged.append(fb)
                    else:
                        prev_start, prev_size = merged[-1]
                        if prev_start + prev_size == fb[0]:
                            merged[-1] = (prev_start, prev_size + fb[1])
                        else:
                            merged.append(fb)
                free_blocks = merged
        if free_blocks:
            max_free_block = max(fb[1] for fb in free_blocks)
        else:
            max_free_block = 0
    return {
        "failed_allocations": failed_allocations,
        "max_free_block": max_free_block,
    }


def find_optimal_max_split_size(trace: list, candidate_sizes: list) -> int:
    best_size = candidate_sizes[0]
    min_failures = float("inf")
    for size in candidate_sizes:
        res = simulate_fragmentation(trace, size)
        if res["failed_allocations"] < min_failures:
            min_failures = res["failed_allocations"]
            best_size = size
    return best_size
