def compute_capture_sizes(max_batch_size: int, num_speculative_tokens: int) -> list:
    sizes = set()
    b = 1
    while b <= max_batch_size:
        for s in range(num_speculative_tokens + 1):
            sizes.add(b * (s + 1))
        b *= 2
    if max_batch_size not in [2**i for i in range(20)] and max_batch_size > 0:
        for s in range(num_speculative_tokens + 1):
            sizes.add(max_batch_size * (s + 1))
    return sorted(list(sizes))
