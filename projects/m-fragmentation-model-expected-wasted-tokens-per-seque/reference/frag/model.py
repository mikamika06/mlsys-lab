def expected_wasted_tokens(length_histogram, block_size):
    total_seqs = sum(length_histogram.values())
    if total_seqs == 0:
        return 0.0
    total_wasted = 0
    for length, count in length_histogram.items():
        rem = length % block_size
        wasted = 0 if rem == 0 else block_size - rem
        total_wasted += wasted * count
    return float(total_wasted) / float(total_seqs)


def optimal_block_size(length_histogram, candidate_block_sizes, bytes_per_token):
    best_size = candidate_block_sizes[0]
    best_cost = float("inf")
    for b in candidate_block_sizes:
        w = expected_wasted_tokens(length_histogram, b)
        avg_len = sum(l * c for l, c in length_histogram.items()) / sum(length_histogram.values())
        cost = (avg_len + w) * bytes_per_token
        if cost < best_cost:
            best_cost = cost
            best_size = b
    return best_size
