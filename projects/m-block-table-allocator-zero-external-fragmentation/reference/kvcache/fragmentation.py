def optimal_block_size(sequence_lengths: list[int], max_b: int = 128) -> int:
    best_b = 1
    min_frag = float('inf')
    for b in range(1, max_b + 1):
        frag = sum(((l + b - 1) // b) * b - l for l in sequence_lengths)
        if frag < min_frag:
            min_frag = frag
            best_b = b
    return best_b
