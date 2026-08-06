def test_optimizer_finds_minimum():
    from optimizer.chunking import optimize_chunk_size, calculate_prefix_savings, benchmark_serving_frontier

    trace = [[1, 2, 3, 4, 5, 6, 7, 8], [1, 2, 3, 4, 9, 10, 11, 12]]
    sizes = [2, 4, 8]

    costs = []
    for s in sizes:
        sv = calculate_prefix_savings(trace, s)
        costs.append(benchmark_serving_frontier(trace, s, sv))

    expected_idx = min(range(len(costs)), key=costs.__getitem__)
    actual_idx = optimize_chunk_size(trace, sizes)

    assert actual_idx == expected_idx
