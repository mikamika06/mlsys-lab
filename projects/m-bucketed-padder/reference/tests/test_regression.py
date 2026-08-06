from padder.ladder import find_optimal_ladder


def test_ladder_alignment():
    lengths = [10, 23, 45, 61]
    candidate_bounds = [16, 30, 48, 61, 64, 80]
    alignment = 8
    max_buckets = 3
    compilation_cost = 10.0

    ladder, _ = find_optimal_ladder(
        lengths, candidate_bounds, max_buckets, compilation_cost, alignment=alignment
    )

    for bound in ladder:
        assert bound % alignment == 0, f"Bound {bound} not aligned to {alignment}"
