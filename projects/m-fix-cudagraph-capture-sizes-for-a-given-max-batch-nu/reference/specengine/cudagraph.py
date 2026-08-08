def get_capture_sizes(max_batch_size: int, num_speculative_tokens: int) -> list:
    sizes = []
    curr = 1
    while curr <= max_batch_size:
        sizes.append(curr)
        if curr < 4:
            curr += 1
        elif curr < 16:
            curr += 2
        elif curr < 64:
            curr += 4
        else:
            curr += 8
    if max_batch_size not in sizes:
        sizes.append(max_batch_size)
        sizes = sorted(list(set(sizes)))
    return sizes
