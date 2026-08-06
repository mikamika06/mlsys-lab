def compute_capture_sizes(max_batch_size: int, num_speculative_tokens: int) -> list:
    base_sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
    effective_max = max_batch_size * (num_speculative_tokens + 1)
    sizes = []
    for b in base_sizes:
        if b <= effective_max:
            sizes.append(b)
    if effective_max not in sizes:
        sizes.append(effective_max)
    return sorted(list(set(sizes)))
