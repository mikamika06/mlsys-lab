import numpy as np


def generate_lengths(seed=42, n=500, max_len=256):
    rng = np.random.RandomState(seed)
    lengths = rng.geometric(p=0.03, size=n)
    return np.clip(lengths, 1, max_len).tolist()


def ref_assign_bucket(length, bucket_boundaries):
    for b in sorted(bucket_boundaries):
        if length <= b:
            return b
    return max(bucket_boundaries)


def ref_pad_batch(sequences, bucket_boundaries, pad_val=0):
    max_seq_len = max(len(seq) for seq in sequences)
    chosen_bucket = ref_assign_bucket(max_seq_len, bucket_boundaries)
    batch_size = len(sequences)
    padded = np.full((batch_size, chosen_bucket), pad_val, dtype=np.int64)
    mask = np.zeros((batch_size, chosen_bucket), dtype=np.int64)
    for i, seq in enumerate(sequences):
        seq_len = min(len(seq), chosen_bucket)
        padded[i, :seq_len] = seq[:seq_len]
        mask[i, :seq_len] = 1
    return padded, mask, chosen_bucket


def ref_compute_padding_waste(lengths, bucket_boundaries):
    sorted_bounds = sorted(bucket_boundaries)
    total_unpadded = sum(lengths)
    total_padded = 0
    for l in lengths:
        b = ref_assign_bucket(l, sorted_bounds)
        total_padded += b
    waste_tokens = total_padded - total_unpadded
    ratio = waste_tokens / float(total_padded) if total_padded > 0 else 0.0
    return waste_tokens, ratio


def ref_find_optimal_ladder(lengths, candidate_bounds, max_buckets, compilation_cost, alignment=1):
    candidates = sorted(list({b for b in candidate_bounds if b % alignment == 0 and b >= max(lengths)}))
    if not candidates:
        max_l = max(lengths)
        rem = max_l % alignment
        candidates = [max_l if rem == 0 else max_l + (alignment - rem)]

    n = len(lengths)
    valid_candidates = sorted(candidates)

    best_cost = float("inf")
    best_ladder = []

    def evaluate_ladder(ladder):
        nonlocal best_cost, best_ladder
        w_tokens, _ = ref_compute_padding_waste(lengths, ladder)
        cost = w_tokens + len(ladder) * compilation_cost
        if cost < best_cost or (cost == best_cost and len(ladder) < len(best_ladder)):
            best_cost = cost
            best_ladder = sorted(ladder)

    import itertools
    for k in range(1, max_buckets + 1):
        for combo in itertools.combinations(valid_candidates, k):
            if max(combo) >= max(lengths):
                evaluate_ladder(list(combo))

    return best_ladder, best_cost
