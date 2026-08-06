import itertools
from padder.cost import compute_padding_waste


def find_optimal_ladder(lengths, candidate_bounds, max_buckets, compilation_cost, alignment=1):
    valid_candidates = sorted(list({
        b for b in candidate_bounds
        if b % alignment == 0 and b >= max(lengths)
    }))
    if not valid_candidates:
        max_l = max(lengths)
        rem = max_l % alignment
        valid_candidates = [max_l if rem == 0 else max_l + (alignment - rem)]

    best_cost = float("inf")
    best_ladder = []

    for k in range(1, max_buckets + 1):
        for combo in itertools.combinations(valid_candidates, k):
            ladder = list(combo)
            if max(ladder) < max(lengths):
                continue
            waste, _ = compute_padding_waste(lengths, ladder)
            cost = waste + len(ladder) * compilation_cost
            if cost < best_cost or (cost == best_cost and len(ladder) < len(best_ladder)):
                best_cost = cost
                best_ladder = sorted(ladder)

    return best_ladder, best_cost
