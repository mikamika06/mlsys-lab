def compute_pair_counts(assignments, num_blocks, block_size, causal=True):
    counts = []
    for blocks in assignments:
        pairs = 0
        for b in blocks:
            for o in range(num_blocks):
                if not causal or b >= o:
                    pairs += block_size * block_size
                elif b == o:
                    pairs += block_size * block_size
        counts.append(pairs)
    return counts


def imbalance_ratio(counts):
    if not counts or sum(counts) == 0:
        return 0.0
    mean_val = sum(counts) / len(counts)
    if mean_val == 0:
        return 0.0
    max_val = max(counts)
    return max_val / mean_val
